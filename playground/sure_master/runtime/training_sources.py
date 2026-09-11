"""Small checked edits to workspace copies of upstream training code."""

from __future__ import annotations
import ast
from pathlib import Path


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(
            f"Unsupported upstream trainer version: expected one occurrence of {old[:75]!r}"
        )
    return text.replace(old, new, 1)


def prepare_f5_training_source(root: Path) -> None:
    path = root / "src/f5_tts/model/trainer.py"
    text = path.read_text()
    if "self.sure_training.configure_loader" in text:
        return
    text = replace_once(
        text,
        "        start_update = self.load_checkpoint()",
        "        self.sure_training.configure_loader(train_dataloader)\n        start_update = self.load_checkpoint()",
    )
    start = text.index(
        "        if exists(resumable_with_seed):\n            orig_epoch_step"
    )
    end = text.index("        for epoch in range(skipped_epoch, self.epochs):", start)
    text = (
        text[:start]
        + """        skipped_epoch = self.sure_training.progress['epoch']
        skipped_batch = self.sure_training.progress['batch']
        skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)

"""
        + text[end:]
    )
    text = replace_once(
        text,
        "            for batch in current_dataloader:",
        "            for batch_index, batch in self.sure_training.batches_for_epoch(current_dataloader, epoch):\n                self.sure_training.before_forward()",
    )
    text = replace_once(
        text,
        "                    self.accelerator.backward(loss)",
        '                    if not torch.isfinite(loss).all():\n                        raise FloatingPointError("Nonfinite F5 training loss")\n                    self.accelerator.backward(loss)',
    )
    text = replace_once(
        text,
        "                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)",
        '                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)\n                        if not torch.isfinite(grad_norm).all():\n                            raise FloatingPointError("Nonfinite F5 gradients")',
    )
    text = replace_once(
        text,
        "                    global_update += 1",
        "                    global_update += 1\n                    self.sure_training.after_update(epoch, batch_index + 1, global_update)",
    )
    text = replace_once(
        text,
        "        self.save_checkpoint(global_update, last=True)\n\n        self.accelerator.end_training()",
        "            self.sure_training.epoch_end(epoch + 1)\n\n        self.save_checkpoint(global_update, last=True)\n\n        self.accelerator.end_training()",
    )
    ast.parse(text)
    path.write_text(text)
    prepare_csv = root / "src/f5_tts/train/datasets/prepare_csv_wavs.py"
    if prepare_csv.exists():
        source = prepare_csv.read_text()
        marker = "PRETRAINED_VOCAB_PATH = files(\"f5_tts\").joinpath(\"../../data/Emilia_ZH_EN_pinyin/vocab.txt\")"
        replacement = (
            "PRETRAINED_VOCAB_PATH = Path(os.environ.get(\"SURE_F5_VOCAB_FILE\", "
            "str(files(\"f5_tts\").joinpath(\"../../data/Emilia_ZH_EN_pinyin/vocab.txt\"))))"
        )
        if marker in source and replacement not in source:
            prepare_csv.write_text(source.replace(marker, replacement, 1))
