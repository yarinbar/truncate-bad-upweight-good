import unittest

import torch

from qrpo.trainers.qrpo import compute_ours_logit_terms


class ComputeOursLogitTermsTests(unittest.TestCase):
    def test_flag_off_preserves_raw_logratio_scores(self):
        _, _, right_term_chosen, right_term_rejected = compute_ours_logit_terms(
            chosen_logps=torch.tensor([10.0]),
            rejected_logps=torch.tensor([8.0]),
            ref_chosen_logps=torch.tensor([4.0]),
            ref_rejected_logps=torch.tensor([6.0]),
            chosen_lengths=torch.tensor([3]),
            rejected_lengths=torch.tensor([5]),
            beta=0.5,
            threshold=torch.tensor(1.25),
            use_length_normalized_logratios=False,
        )

        self.assertTrue(torch.allclose(right_term_chosen, torch.tensor([4.25])))
        self.assertTrue(torch.allclose(right_term_rejected, torch.tensor([2.25])))

    def test_flag_on_divides_raw_logratio_by_completion_length(self):
        _, _, right_term_chosen, right_term_rejected = compute_ours_logit_terms(
            chosen_logps=torch.tensor([10.0]),
            rejected_logps=torch.tensor([8.0]),
            ref_chosen_logps=torch.tensor([4.0]),
            ref_rejected_logps=torch.tensor([6.0]),
            chosen_lengths=torch.tensor([3]),
            rejected_lengths=torch.tensor([5]),
            beta=0.5,
            threshold=torch.tensor(1.25),
            use_length_normalized_logratios=True,
        )

        self.assertTrue(torch.allclose(right_term_chosen, torch.tensor([2.25])))
        self.assertTrue(torch.allclose(right_term_rejected, torch.tensor([1.45])))

    def test_length_one_is_a_noop(self):
        normalized = compute_ours_logit_terms(
            chosen_logps=torch.tensor([9.0]),
            rejected_logps=torch.tensor([7.0]),
            ref_chosen_logps=torch.tensor([5.0]),
            ref_rejected_logps=torch.tensor([3.0]),
            chosen_lengths=torch.tensor([1]),
            rejected_lengths=torch.tensor([1]),
            beta=0.25,
            threshold=torch.tensor(0.75),
            use_length_normalized_logratios=True,
        )
        raw = compute_ours_logit_terms(
            chosen_logps=torch.tensor([9.0]),
            rejected_logps=torch.tensor([7.0]),
            ref_chosen_logps=torch.tensor([5.0]),
            ref_rejected_logps=torch.tensor([3.0]),
            chosen_lengths=torch.tensor([1]),
            rejected_lengths=torch.tensor([1]),
            beta=0.25,
            threshold=torch.tensor(0.75),
            use_length_normalized_logratios=False,
        )

        for normalized_tensor, raw_tensor in zip(normalized, raw):
            self.assertTrue(torch.allclose(normalized_tensor, raw_tensor))

    def test_zero_length_raises_clear_error(self):
        with self.assertRaisesRegex(
            ValueError, "Completion lengths must be positive"
        ):
            compute_ours_logit_terms(
                chosen_logps=torch.tensor([9.0]),
                rejected_logps=torch.tensor([7.0]),
                ref_chosen_logps=torch.tensor([5.0]),
                ref_rejected_logps=torch.tensor([3.0]),
                chosen_lengths=torch.tensor([0]),
                rejected_lengths=torch.tensor([1]),
                beta=0.25,
                threshold=torch.tensor(0.75),
                use_length_normalized_logratios=True,
            )


if __name__ == "__main__":
    unittest.main()
