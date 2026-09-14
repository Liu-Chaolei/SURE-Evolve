"""Download the fixed vocoder and CPU transcription resources before model jobs."""

from __future__ import annotations

import argparse
from pathlib import Path

from playground.sure_master.core.artifacts import file_digest
from playground.sure_master.runtime.training_state import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    from modelscope import snapshot_download as modelscope_download

    args.output.mkdir(parents=True, exist_ok=True)
    vocoder = args.output / "vocos-mel-24khz"
    snapshot_download(
        "charactr/vocos-mel-24khz",
        revision="0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21",
        local_dir=vocoder,
        allow_patterns=["config.yaml", "pytorch_model.bin"],
    )
    model_id = (
        "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    )
    paraformer = args.output / "metric_cache/modelscope/models" / model_id
    modelscope_download(model_id, local_dir=str(paraformer))
    files = [
        vocoder / "config.yaml",
        vocoder / "pytorch_model.bin",
        paraformer / "configuration.json",
        paraformer / "model.pt",
    ]
    atomic_json(
        args.output / "resources.json",
        {
            "vocoder": str(vocoder.resolve()),
            "paraformer": str(paraformer.resolve()),
            "files": {str(p.resolve()): file_digest(p) for p in files},
        },
    )


if __name__ == "__main__":
    main()
