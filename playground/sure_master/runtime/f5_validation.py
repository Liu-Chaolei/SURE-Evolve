"""Fixed validation-loss diagnostics which do not change training RNG or selection."""

from __future__ import annotations

import json
from pathlib import Path
import random

from .training_state import capture_rng, restore_rng


class F5Validation:
    def __init__(self, trainer, manifest: Path, mel: dict, backend: str):
        from datasets import Dataset
        from f5_tts.model.dataset import CustomDataset, collate_fn
        from f5_tts.model.utils import convert_char_to_pinyin
        from torch.utils.data import DataLoader, Subset

        rows = [
            json.loads(line)
            for line in manifest.read_text().splitlines()
            if line.strip()
        ]
        rows.sort(key=lambda row: row["sample_id"])
        random.Random(42).shuffle(rows)
        rows = rows[:256]
        if not rows:
            raise ValueError("Empty F5 training validation split")
        texts = convert_char_to_pinyin([row["text"] for row in rows], polyphone=True)
        raw = Dataset.from_list(
            [
                {"audio_path": row["audio"], "text": text, "duration": row["duration"]}
                for row, text in zip(rows, texts)
            ]
        )
        data = CustomDataset(
            raw, durations=[r["duration"] for r in rows], preprocessed_mel=False, **mel
        )
        accelerator = trainer.accelerator
        indices = list(
            range(accelerator.process_index, len(rows), accelerator.num_processes)
        )
        self.loader = DataLoader(
            Subset(data, indices), batch_size=1, collate_fn=collate_fn, num_workers=0
        )
        self.trainer, self.backend = trainer, backend

    def __call__(self) -> dict:
        import numpy as np
        import torch

        accelerator = self.trainer.accelerator
        model = accelerator.unwrap_model(self.trainer.model)
        state, mode = capture_rng(self.backend), model.training
        seed = 1042 + accelerator.process_index
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.backend != "cpu":
            getattr(torch, self.backend).manual_seed(seed)
        model.eval()
        total, count = 0.0, 0
        try:
            with torch.no_grad():
                for batch in self.loader:
                    loss = model(
                        batch["mel"].to(accelerator.device).permute(0, 2, 1),
                        text=batch["text"],
                        lens=batch["mel_lengths"].to(accelerator.device),
                    )[0]
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite F5 validation loss")
                    total += float(loss)
                    count += 1
            sums = (
                accelerator.reduce(
                    torch.tensor([total, count], device=accelerator.device),
                    reduction="sum",
                )
                .cpu()
                .tolist()
            )
            return {"loss": sums[0] / sums[1], "samples": int(sums[1])}
        finally:
            model.train(mode)
            restore_rng(state, self.backend)
