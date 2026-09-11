"""Real multi-NPU Zipformer update, synchronization and checkpoint probe."""

from __future__ import annotations
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


def rank_probe(rank: int, world_size: int, recipe: str, icefall: str, output: str):
    import torch
    import torch_npu  # noqa: F401
    import k2
    from playground.sure_master.runtime.distributed import (
        setup_dist,
        cleanup_dist,
        backend_digest,
    )

    setup_dist(rank, world_size, 29671)
    torch.set_num_threads(8)
    sys.path[:0] = [recipe, icefall]
    spec = importlib.util.spec_from_file_location("train", Path(recipe) / "train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train"] = module
    spec.loader.exec_module(module)
    torch.manual_seed(42)
    params = module.get_params()
    params.update(vars(module.get_parser().parse_args([])))
    params.vocab_size, params.blank_id = 32, 0
    model = module.get_model(params).to(f"npu:{rank}")
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], find_unused_parameters=True
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    torch.manual_seed(42 + rank)
    x = torch.randn(2, 80, 80, device=f"npu:{rank}")
    lengths = torch.tensor([80, 72], device=f"npu:{rank}")
    labels = k2.RaggedTensor([[1, 2, 3], [4, 5]])
    model.train()
    simple, pruned = model(x, lengths, labels)
    loss = simple.sum() + pruned.sum()
    loss.backward()
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters or not all(
        torch.isfinite(p.grad).all().item() for p in parameters
    ):
        raise RuntimeError("Nonfinite/missing DDP gradients")
    chosen = next(p for p in parameters if p.grad.abs().max().item() > 0)
    before = chosen.detach().clone()
    optimizer.step()
    if torch.equal(before, chosen):
        raise RuntimeError("No weight update")
    signature = torch.stack([p.detach().float().sum() for p in model.parameters()])
    signatures = [torch.empty_like(signature) for _ in range(world_size)]
    torch.distributed.all_gather(signatures, signature)
    for other in signatures:
        torch.testing.assert_close(signature, other, atol=1e-4, rtol=1e-5)
    checkpoint = Path(output) / "ddp-checkpoint.pt"
    if rank == 0:
        torch.save(
            {"model": model.module.state_dict(), "optimizer": optimizer.state_dict()},
            checkpoint,
        )
    torch.distributed.barrier()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.module.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    if rank == 0:
        (Path(output) / "probe_result.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "backend_digest": backend_digest(),
                    "world_size": world_size,
                    "backend": "hccl",
                    "loss": loss.item(),
                    "gradient_tensors": len(parameters),
                    "ranks_synchronized": True,
                    "checkpoint_reloaded": True,
                },
                indent=2,
            )
            + "\n"
        )
    cleanup_dist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icefall", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, choices=(2, 8), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        SURE_ASR_RECIPE_PROFILE="tedlium3_zipformer_native",
        SURE_USE_FP16="0",
        SURE_CPU_THREADS="8",
    )
    from playground.sure_master.runtime.icefall import prepare_npu_recipe

    recipe = prepare_npu_recipe(
        args.icefall / "egs/tedlium3/ASR/zipformer", args.output / "recipe"
    )
    import torch.multiprocessing as mp

    mp.spawn(
        rank_probe,
        args=(args.world_size, str(recipe), str(args.icefall), str(args.output)),
        nprocs=args.world_size,
    )


if __name__ == "__main__":
    main()
