"""CPU audio I/O independent of the Torch/torchaudio codec ABI."""

from __future__ import annotations

from types import SimpleNamespace


def load_audio(path):
    """Read the task's WAV input as a channels-first float32 Torch tensor."""
    import soundfile as sf
    import torch

    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(samples.T.copy()), sample_rate


def audio_info(path):
    import soundfile as sf

    info = sf.info(path)
    return SimpleNamespace(
        sample_rate=info.samplerate, num_frames=info.frames, num_channels=info.channels
    )
