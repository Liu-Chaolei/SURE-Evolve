"""Read-only CPU checkpoint validation inside an owned Slurm container."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    import torch
    candidates = sorted((args.workspace / "models/zipformer_candidate").glob("epoch-*.pt"),
                        key=lambda p: int(p.stem.split("-")[-1]), reverse=True)
    for path in candidates:
        before = path.stat()
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            epoch = int(path.stem.split("-")[-1])
            if int(checkpoint.get("cur_epoch", 0)) != epoch or not all(
                    k in checkpoint for k in ("model", "optimizer", "scheduler", "batch_idx_train")):
                continue
            progress = int(checkpoint["batch_idx_train"])
            if progress <= 0:
                continue
            del checkpoint
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                continue
            print("\nASR_CHECKPOINT_JSON=" + json.dumps({"checkpoint": str(path), "epoch": epoch, "next_epoch": epoch + 1,
                              "batch_idx_train": progress, "sha256": digest.hexdigest()}), flush=True)
            return
        except (OSError, RuntimeError, EOFError, ValueError):
            continue
    raise SystemExit("No complete resumable epoch checkpoint")


if __name__ == "__main__":
    main()
