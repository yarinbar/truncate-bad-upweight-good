import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from qrpo.evals import run_online_eval as online_eval
from qrpo.evals.run_online_eval import (
    HH_IDENTITY_CHAT_TEMPLATE,
    _balanced_eval_split_subsample_indices,
    _discover_checkpoints,
    _ensure_hh_chat_template,
    _load_or_create_eval_split_subsample_indices,
    _gather_label_log_probs,
    _get_message_content,
    _prompt_from_ref_dataset_row,
    _results_dataset_leaf_name,
    _scores_from_model_output,
    _subset_positions_for_sampled_indices,
    _tokenize_reward_batch,
)


class OnlineEvalCheckpointDiscoveryTests(unittest.TestCase):
    def test_discover_checkpoints_returns_only_requested_step(self):
        with TemporaryDirectory() as tmpdir:
            training_dir = Path(tmpdir)
            for step in (0, 160, 320, 480):
                (training_dir / f"checkpoint-{step}").mkdir()

            checkpoints = _discover_checkpoints(training_dir, [480])

            self.assertEqual(checkpoints, [training_dir / "checkpoint-480"])

    def test_discover_checkpoints_without_filter_returns_all_in_numeric_order(self):
        with TemporaryDirectory() as tmpdir:
            training_dir = Path(tmpdir)
            for step in (480, 0, 160):
                (training_dir / f"checkpoint-{step}").mkdir()

            checkpoints = _discover_checkpoints(training_dir, [])

            self.assertEqual(
                checkpoints,
                [
                    training_dir / "checkpoint-0",
                    training_dir / "checkpoint-160",
                    training_dir / "checkpoint-480",
                ],
            )


class GatherLabelLogProbsTests(unittest.TestCase):
    def test_matches_log_softmax_gather(self):
        logits = torch.randn(2, 3, 7, dtype=torch.float32)
        labels = torch.tensor([[1, 2, 3], [0, 4, 6]])

        expected = torch.log_softmax(logits, dim=-1).gather(
            2, labels.unsqueeze(-1)
        ).squeeze(-1)

        actual = _gather_label_log_probs(logits, labels)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))


class FakeTokenizer:
    def __init__(self, *, chat_template=None):
        self.chat_template = chat_template
        self.calls = []

    def __call__(self, *args, **kwargs):
        if "return_dict" in kwargs:
            raise TypeError(
                "PreTrainedTokenizerFast._batch_encode_plus() got an unexpected "
                "keyword argument 'return_dict'"
            )
        self.calls.append((args, kwargs))
        batch_len = len(args[0])
        return {
            "input_ids": torch.arange(batch_len * 4).reshape(batch_len, 4),
            "attention_mask": torch.ones(batch_len, 4, dtype=torch.long),
        }

    def apply_chat_template(self, chats, **kwargs):
        self.calls.append((("chat", chats), kwargs))
        if kwargs.get("tokenize", True) is False:
            return "CHAT:" + chats[0]["content"] + ":GEN"
        batch_len = len(chats)
        return {
            "input_ids": torch.arange(batch_len * 5).reshape(batch_len, 5),
            "attention_mask": torch.ones(batch_len, 5, dtype=torch.long),
        }


class PythiaOnlineEvalCompatibilityTests(unittest.TestCase):
    def test_prompt_from_ref_dataset_row_supports_string_and_chat_preferences(self):
        self.assertEqual(
            _prompt_from_ref_dataset_row({"prompt": "\n\nHuman: hi\n\nAssistant:"}),
            "\n\nHuman: hi\n\nAssistant:",
        )
        self.assertEqual(
            _prompt_from_ref_dataset_row(
                {"chosen": [{"role": "user", "content": "chat prompt"}]}
            ),
            "chat prompt",
        )

    def test_ensure_hh_chat_template_installs_identity_template_when_missing(self):
        tokenizer = FakeTokenizer(chat_template=None)

        returned = _ensure_hh_chat_template(tokenizer)

        self.assertEqual(tokenizer.chat_template, HH_IDENTITY_CHAT_TEMPLATE)
        self.assertIs(returned, tokenizer)

    def test_ensure_hh_chat_template_preserves_existing_template(self):
        tokenizer = FakeTokenizer(chat_template="template")

        _ensure_hh_chat_template(tokenizer)

        self.assertEqual(tokenizer.chat_template, "template")

    def test_reward_tokenization_uses_prompt_completion_pairs_without_chat_template(self):
        tokenizer = FakeTokenizer(chat_template=None)
        batch = [
            [
                {"role": "user", "content": "prompt A"},
                {"role": "assistant", "content": "completion A"},
            ],
            [
                {"role": "user", "content": "prompt B"},
                {"role": "assistant", "content": "completion B"},
            ],
        ]

        tokens = _tokenize_reward_batch(tokenizer, batch, max_length=128)

        self.assertEqual(
            tokenizer.calls[0][0],
            (["prompt A", "prompt B"], ["completion A", "completion B"]),
        )
        self.assertTrue(
            torch.equal(tokens["attention_mask"], torch.ones(2, 4, dtype=torch.long))
        )

    def test_scores_from_model_output_supports_logits(self):
        output = SimpleNamespace(logits=torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

        self.assertEqual(_scores_from_model_output(output), [2.0, 4.0])

    def test_scores_from_model_output_preserves_1d_reward_scores(self):
        output = SimpleNamespace(score=torch.tensor([1.0, 2.0, 3.0]))

        self.assertEqual(_scores_from_model_output(output), [1.0, 2.0, 3.0])

    def test_validate_reward_output_lengths_reports_mismatched_rewards(self):
        validator = getattr(
            online_eval,
            "_validate_reward_output_lengths",
            lambda **kwargs: None,
        )

        with self.assertRaisesRegex(
            ValueError,
            "reward count 2 != completions count 3",
        ):
            validator(
                completions_len=3,
                all_rewards=[0.1, 0.2],
                all_reward_tokens=[[1], [2], [3]],
                all_reward_tokens_len=[1, 1, 1],
                all_reward_texts=["a", "b", "c"],
            )

    def test_get_message_content_supports_chat_and_raw_text(self):
        self.assertEqual(
            _get_message_content({"role": "user", "content": "chat prompt"}),
            "chat prompt",
        )
        self.assertEqual(_get_message_content("raw prompt"), "raw prompt")

    def test_balanced_eval_split_subsample_indices_are_deterministic(self):
        selection_indices = np.arange(0, 10)
        reporting_indices = np.arange(100, 110)

        first = _balanced_eval_split_subsample_indices(
            selection_indices,
            reporting_indices,
            subsample_size=8,
            seed=123,
        )
        second = _balanced_eval_split_subsample_indices(
            selection_indices,
            reporting_indices,
            subsample_size=8,
            seed=123,
        )

        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(first), 8)
        self.assertEqual(len(set(first).intersection(selection_indices)), 4)
        self.assertEqual(len(set(first).intersection(reporting_indices)), 4)

    def test_subset_positions_for_sampled_indices_remap_original_eval_keys(self):
        sampled_indices = np.array([2, 4, 101, 105])
        selection_indices = np.array([1, 2, 3, 4])
        reporting_indices = np.array([100, 101, 105])

        selection_positions, reporting_positions, aggregate_positions = (
            _subset_positions_for_sampled_indices(
                sampled_indices,
                selection_indices,
                reporting_indices,
            )
        )

        self.assertTrue(np.array_equal(selection_positions, np.array([0, 1])))
        self.assertTrue(np.array_equal(reporting_positions, np.array([2, 3])))
        self.assertTrue(
            np.array_equal(aggregate_positions, np.array([0, 1, 2, 3]))
        )

    def test_results_dataset_leaf_name_uses_rewards_dataset_when_kl_is_skipped(self):
        self.assertEqual(
            _results_dataset_leaf_name(SimpleNamespace(skip_kl=True)),
            "rewards-dataset",
        )
        self.assertEqual(
            _results_dataset_leaf_name(SimpleNamespace(skip_kl=False)),
            "kl-dataset",
        )
        self.assertEqual(_results_dataset_leaf_name(SimpleNamespace()), "kl-dataset")

    def test_eval_split_subsample_indices_are_cached_in_shared_keys_dir(self):
        with TemporaryDirectory() as tmpdir:
            dataset_path = (
                Path(tmpdir)
                / "data/shared/datasets-with-ref-rewards/merged"
                / "pythia-sft-hh-oarm-temp1-ref50-offpolicy2best-oarm"
            )
            config = SimpleNamespace(
                dataset_args=SimpleNamespace(dataset_path=str(dataset_path))
            )
            selection_indices = np.arange(0, 10)
            reporting_indices = np.arange(100, 110)

            first = _load_or_create_eval_split_subsample_indices(
                config,
                selection_indices,
                reporting_indices,
                subsample_size=8,
                seed=123,
            )
            cache_path = (
                Path(tmpdir)
                / "data/shared/datasets-with-chosen-rewards/merged/hh-oarm"
                / "eval-subsamples/subsample-size8-seed123-indices.npy"
            )
            cached_override = np.array([7, 8, 9])
            np.save(cache_path, cached_override)

            second = _load_or_create_eval_split_subsample_indices(
                config,
                selection_indices,
                reporting_indices,
                subsample_size=8,
                seed=123,
            )

            self.assertTrue(cache_path.exists())
            self.assertEqual(len(first), 8)
            self.assertTrue(np.array_equal(second, cached_override))


if __name__ == "__main__":
    unittest.main()
