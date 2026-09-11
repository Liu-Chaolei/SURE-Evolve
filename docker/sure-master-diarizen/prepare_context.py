"""Prepare a small build context without model weights, datasets or secrets."""
import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise SystemExit("Use a new context directory")
args.output.mkdir(parents=True)
for name in ("Dockerfile", "requirements.txt", "constraints.txt", "md-eval-22.pl"):
    shutil.copy2(Path(__file__).parent / name, args.output / name)
shutil.copytree(args.source, args.output / "source", ignore=shutil.ignore_patterns(
    ".git", ".env", ".env.*", "checkpoints", "data", "__pycache__", "*.log", "runs"))
