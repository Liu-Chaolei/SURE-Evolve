"""Fixed native Zipformer baseline, using the same execution path as candidates."""

import os
import subprocess
from pathlib import Path


def main():
    assert Path("base_model/recipe/train.py").is_file()
    subprocess.run(
        [
            os.environ["SURE_ICEFALL_PYTHON"],
            os.environ["SURE_ASR_ZIPFORMER_WRAPPER"],
            "--action",
            "train_decode",
            "--candidate-type",
            "fine_tune",
        ],
        check=True,
    )
    assert Path("artifacts/hyp.txt").is_file()


if __name__ == "__main__":
    main()
