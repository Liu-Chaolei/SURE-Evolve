"""Run frozen Icefall decoding with the Ascend PackedSequence correction.

The patch lives only in the decoding subprocess, including for projects that
were prepared before this runtime repair. Frozen scientific source stays intact.
"""
from pathlib import Path
import runpy
import sys

import torch

from asr_runtime.npu_rnn import pack_padded_sequence


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m asr_runtime.npu_decode DECODE_SCRIPT [ARG ...]")
    script = Path(sys.argv[1]).resolve(strict=True)
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path.insert(0, str(script.parent))
    torch.nn.utils.rnn.pack_padded_sequence = pack_padded_sequence
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
