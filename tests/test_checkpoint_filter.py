import unittest
from pathlib import Path

from qrpo.evals.checkpoint_filter import (
    normalize_checkpoint_steps,
    select_checkpoint_paths,
)


class CheckpointFilterTest(unittest.TestCase):
    def test_normalize_checkpoint_steps_returns_empty_for_no_filter(self):
        self.assertEqual(normalize_checkpoint_steps(None), [])
        self.assertEqual(normalize_checkpoint_steps([]), [])

    def test_normalize_checkpoint_steps_deduplicates_and_sorts(self):
        self.assertEqual(
            normalize_checkpoint_steps(["480", 160, "480"]),
            [160, 480],
        )

    def test_normalize_checkpoint_steps_rejects_invalid_values(self):
        for value in (0, -1, "abc", "1.5", True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    normalize_checkpoint_steps([value])

    def test_select_checkpoint_paths_without_filter_preserves_order(self):
        checkpoints = (
            Path("/runs/checkpoint-480"),
            Path("/runs/not-a-checkpoint"),
            Path("/runs/checkpoint-160"),
        )

        self.assertEqual(
            select_checkpoint_paths(checkpoints, [], context=Path("/runs")),
            list(checkpoints),
        )

    def test_select_checkpoint_paths_returns_requested_steps_in_numeric_order(self):
        checkpoints = [
            Path("/runs/checkpoint-764"),
            Path("/runs/checkpoint-160-extra"),
            Path("/runs/checkpoint-480"),
            Path("/runs/prefix-checkpoint-160"),
            Path("/runs/checkpoint-160"),
        ]

        selected = select_checkpoint_paths(
            checkpoints,
            [480, 160],
            context=Path("/runs"),
        )

        self.assertEqual(
            selected,
            [
                Path("/runs/checkpoint-160"),
                Path("/runs/checkpoint-480"),
            ],
        )

    def test_select_checkpoint_paths_reports_all_missing_steps_and_context(self):
        context = Path("/runs/beta-0.01")

        with self.assertRaises(ValueError) as exc_info:
            select_checkpoint_paths(
                [Path("/runs/beta-0.01/checkpoint-160")],
                [999, 480],
                context=context,
            )

        message = str(exc_info.exception)
        self.assertIn("480", message)
        self.assertIn("999", message)
        self.assertIn(str(context), message)


if __name__ == "__main__":
    unittest.main()
