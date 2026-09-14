"""Regression for the unequal-length Ascend packing failure observed in TEDLIUM."""
import unittest

import torch
from torch.nn.utils.rnn import pack_padded_sequence as native_pack

from playground.sure_master.runtime.npu_rnn import pack_padded_sequence


@unittest.skipUnless(hasattr(torch, "npu") and torch.npu.is_available(), "requires Slurm-allocated NPU")
class PackedSequenceRegression(unittest.TestCase):
    def test_lengths_order_and_noncontiguous_layout(self):
        # Actual failing encoder lengths, permuted to exercise order restoration.
        lengths = torch.tensor([190, 216, 209])
        time_major = torch.arange(216 * 3 * 5, dtype=torch.float32).reshape(216, 3, 5)
        for batch_first in (False, True):
            values = time_major.transpose(0, 1) if batch_first else time_major
            expected = native_pack(values, lengths, batch_first=batch_first, enforce_sorted=False)
            actual = pack_padded_sequence(values.to("npu"), lengths,
                                          batch_first=batch_first, enforce_sorted=False)
            self.assertEqual(actual.data.shape[0], 615)
            self.assertEqual(actual.data.device.type, "npu")
            self.assertEqual(actual.batch_sizes.device.type, "cpu")
            torch.testing.assert_close(actual.data.cpu(), expected.data, rtol=0, atol=0)
            torch.testing.assert_close(actual.batch_sizes, expected.batch_sizes, rtol=0, atol=0)
            torch.testing.assert_close(actual.sorted_indices.cpu(), expected.sorted_indices, rtol=0, atol=0)
            torch.testing.assert_close(actual.unsorted_indices.cpu(), expected.unsorted_indices, rtol=0, atol=0)

    def test_padding_never_enters_packed_data(self):
        lengths = [7, 7, 4, 2]
        values = torch.arange(4 * 7 * 3, dtype=torch.float32).reshape(4, 7, 3)
        changed = values.clone()
        for row, length in enumerate(lengths):
            changed[row, length:] = -10000
        original = pack_padded_sequence(values.to("npu"), lengths, batch_first=True)
        altered = pack_padded_sequence(changed.to("npu"), lengths, batch_first=True)
        self.assertEqual(original.data.shape[0], sum(lengths))
        torch.testing.assert_close(original.data.cpu(), altered.data.cpu(), rtol=0, atol=0)

    def test_gradient_follows_valid_frames_only(self):
        lengths = [2, 5, 3]
        reference = torch.arange(30, dtype=torch.float32).reshape(3, 5, 2).requires_grad_()
        candidate = reference.detach().to("npu").requires_grad_()
        expected = native_pack(reference, lengths, batch_first=True, enforce_sorted=False)
        actual = pack_padded_sequence(candidate, lengths, batch_first=True, enforce_sorted=False)
        expected.data.square().sum().backward()
        actual.data.square().sum().backward()
        # NPU square's backward rounds slightly differently from CPU FP32.
        torch.testing.assert_close(candidate.grad.cpu(), reference.grad, rtol=1e-6, atol=1e-6)
        self.assertTrue(torch.equal(candidate.grad.cpu()[reference.grad == 0], reference.grad[reference.grad == 0]))


if __name__ == "__main__":
    unittest.main()
