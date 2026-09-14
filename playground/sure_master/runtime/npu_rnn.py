"""Preserve PyTorch PackedSequence semantics on Ascend during ASR decoding."""
from __future__ import annotations

from typing import List, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence as torch_pack_padded_sequence


def pack_padded_sequence(
    input: Tensor,
    lengths: Union[Tensor, List[int]],
    batch_first: bool = False,
    enforce_sorted: bool = True,
) -> PackedSequence:
    """Pack valid frames on CPU, then return data and indices to the input device.

    torch_npu 2.10 on Ascend 910B produced padded rows and incorrect data for
    unequal sequence lengths. Icefall consumes data according to batch_sizes,
    so this silently shifts encoder frames between utterances during decoding.
    Keep batch_sizes on CPU as required by PackedSequence; .to() moves its data
    and sorting indices and preserves autograd. CPU/CUDA use native PyTorch.
    """
    if input.device.type != "npu":
        return torch_pack_padded_sequence(input, lengths, batch_first, enforce_sorted)
    cpu_lengths = lengths.cpu() if isinstance(lengths, Tensor) else lengths
    packed = torch_pack_padded_sequence(input.cpu(), cpu_lengths, batch_first, enforce_sorted)
    return packed.to(input.device)
