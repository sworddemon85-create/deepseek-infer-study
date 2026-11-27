import unittest

import torch

from model import DeepSeekMini, MiniConfig


class CachedChunkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(1)
        self.model = DeepSeekMini(MiniConfig(num_layers=2)).eval()
        self.tokens = torch.tensor([[1, 7, 3, 11, 5, 13, 2, 17],
                                    [2, 9, 4, 12, 6, 14, 3, 18]])

    @torch.no_grad()
    def test_cached_chunks_match_full_prefill_logits_and_cache(self):
        expected, expected_c, expected_r = self.model(self.tokens)
        for lengths in [(2, 3, 3), (1, 7), (3, 1, 4), (1,) * 8]:
            with self.subTest(lengths=lengths):
                c_cache = r_cache = None
                outputs = []
                start = 0
                for length in lengths:
                    logits, c_cache, r_cache = self.model(
                        self.tokens[:, start:start + length], c_cache, r_cache)
                    outputs.append(logits)
                    start += length
                torch.testing.assert_close(torch.cat(outputs, dim=1), expected,
                                           atol=1e-5, rtol=1e-5)
                for actual, reference in zip(c_cache + r_cache, expected_c + expected_r):
                    torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)

    @torch.no_grad()
    def test_future_tokens_cannot_change_earlier_cached_chunk_logits(self):
        _, c_cache, r_cache = self.model(self.tokens[:, :2])
        suffix = self.tokens[:, 2:].clone()
        original, _, _ = self.model(suffix, c_cache, r_cache)
        suffix[:, -1] = 900
        changed, _, _ = self.model(suffix, c_cache, r_cache)
        torch.testing.assert_close(changed[:, :-1], original[:, :-1],
                                   atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
