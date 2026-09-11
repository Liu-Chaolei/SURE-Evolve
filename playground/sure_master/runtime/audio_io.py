"""CPU audio I/O independent of the Torch/torchaudio codec ABI."""

from __future__ import annotations

from types import SimpleNamespace
from dataclasses import dataclass


@dataclass
class AudioMetaData:
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int = 0
    encoding: str = "PCM_S"


def load_audio(path, *, frame_offset=0, num_frames=-1):
    """Read the task's WAV input as a channels-first float32 Torch tensor."""
    import soundfile as sf
    import torch

    stop = None if num_frames < 0 else frame_offset + num_frames
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True,
                                   start=frame_offset, stop=stop)
    return torch.from_numpy(samples.T.copy()), sample_rate


def audio_info(path):
    import soundfile as sf

    info = sf.info(path)
    return SimpleNamespace(
        sample_rate=info.samplerate, num_frames=info.frames, num_channels=info.channels
    )
