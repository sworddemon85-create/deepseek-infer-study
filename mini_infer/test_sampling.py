import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from generate import sample_token


class SamplingTests(unittest.TestCase):
    def test_unfiltered_sampling_matches_multinomial(self):
        logits = torch.tensor([0.1, 1.2, -0.7, 0.4])
        for seed in range(20):
            expected = int(torch.multinomial(
                F.softmax(logits / 0.8, dim=-1), 1,
                generator=torch.Generator().manual_seed(seed)))
            actual = sample_token(logits, temperature=0.8,
                                  generator=torch.Generator().manual_seed(seed))
            self.assertEqual(actual, expected)

    def test_filtered_sampling_returns_original_token_id(self):
        logits = torch.tensor([-10.0, -5.0, 10.0, 0.0])
        for top_k, top_p in [(1, 1.0), (0, 0.1), (2, 0.1)]:
            self.assertEqual(sample_token(logits, temperature=1.0,
                                         top_k=top_k, top_p=top_p), 2)

    def test_greedy_does_not_call_multinomial(self):
        with patch('torch.multinomial', side_effect=AssertionError('sampling')):
            self.assertEqual(sample_token(torch.tensor([1.0, 3.0, 2.0])), 1)


if __name__ == '__main__':
    unittest.main()
