"""Exercise real Zipformer forward/loss/backward before a long training run."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import random
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icefall", type=Path, required=True)
    parser.add_argument("--backend", choices=("cuda", "npu", "cpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["SURE_ASR_RECIPE_PROFILE"] = "tedlium3_zipformer"
    os.environ["SURE_USE_FP16"] = "0"
    from playground.sure_master.runtime.accelerator import check_accelerator, prepare_npu_recipe
    info = check_accelerator(args.backend)
    import torch
    import k2
    recipe = args.icefall / "egs/tedlium3/ASR/zipformer"
    args.output.mkdir(parents=True, exist_ok=True)
    if args.backend == "npu":
        recipe = prepare_npu_recipe(recipe, args.output / "recipe")
    sys.path[:0] = [str(recipe.resolve()), str(args.icefall.resolve())]
    spec = importlib.util.spec_from_file_location("train", recipe / "train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train"] = module
    spec.loader.exec_module(module)
    params = module.get_params()
    params.update(vars(module.get_parser().parse_args([])))
    params.vocab_size, params.blank_id = 32, 0
    torch.manual_seed(42)
    model = module.get_model(params).to(args.backend)
    model.eval()  # deterministic probe; gradients remain enabled
    x = torch.randn(1, 80, 80, device=args.backend)
    lengths = torch.tensor([80], device=args.backend)
    labels = k2.RaggedTensor([[1, 2, 3]])
    if args.backend == "cuda":
        labels = labels.to("cuda")
    # Balancer/Whiten can modify gradients stochastically even in eval mode.
    # Replay the Python RNG state so CPU and NPU exercise identical branches.
    random.seed(42)
    rng_state = random.getstate()
    simple, pruned = model(x, lengths, labels)
    loss = simple.sum() + pruned.sum()
    loss.backward()
    if args.backend == "npu":
        reference = copy.deepcopy(model).to("cpu")
        reference.zero_grad(set_to_none=True)
        random.setstate(rng_state)
        cpu_simple, cpu_pruned = reference(x.detach().cpu(), lengths.cpu(), labels)
        cpu_loss = cpu_simple.sum() + cpu_pruned.sum()
        cpu_loss.backward()
        torch.testing.assert_close(loss.detach().cpu(), cpu_loss.detach(), rtol=5e-3, atol=5e-3)
        compared = 0
        for (name, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
            if actual.grad is not None and expected.grad is not None:
                torch.testing.assert_close(actual.grad.cpu(), expected.grad, rtol=1e-2, atol=1e-3, msg=lambda message: name + ": " + message)
                compared += 1
        info.update(cpu_reference_loss=float(cpu_loss.item()), compared_gradient_tensors=compared)
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters or not all(torch.isfinite(p.grad).all().item() for p in parameters):
        raise RuntimeError("Zipformer gradients are absent or non-finite")
    parameter = next(p for p in parameters if p.grad.abs().max().item() > 0)
    before = parameter.detach().clone()
    torch.optim.SGD(model.parameters(), lr=1e-3).step()
    if torch.equal(before, parameter):
        raise RuntimeError("Zipformer optimizer did not update weights")
    model.train()
    model.zero_grad(set_to_none=True)
    train_simple, train_pruned = model(x, lengths, labels)
    training_loss = train_simple.sum() + train_pruned.sum()
    training_loss.backward()
    if not torch.isfinite(training_loss).item() or not all(
        torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None
    ):
        raise RuntimeError("Zipformer training-mode forward/backward failed")
    torch.optim.SGD(model.parameters(), lr=1e-3).step()
    info.update(status="passed", loss=float(loss.item()), gradient_tensors=len(parameters),
                training_mode_loss=float(training_loss.item()), training_mode_backward=True,
                parameter_update=True, parameters=sum(p.numel() for p in model.parameters()))
    (args.output / "probe_result.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
