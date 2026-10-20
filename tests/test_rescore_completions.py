import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from rescore_completions import build_eval_cmd, discover_training_runs


class DiscoverTrainingRunsTests(unittest.TestCase):
    def _create_training_run(self, root):
        hash_dir = Path(root) / "example-run" / "checkpoints" / "hash"
        config_path = hash_dir / "config" / "process-0" / "config-postprocessed.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(yaml.safe_dump({"example": "config"}))
        for step in (0, 160, 480):
            (hash_dir / f"checkpoint-{step}").mkdir()
        return hash_dir

    def test_discover_training_runs_selects_requested_checkpoint(self):
        with TemporaryDirectory() as tmp:
            self._create_training_run(tmp)

            runs = discover_training_runs(Path(tmp), checkpoint_steps=[480])

        self.assertEqual(
            [checkpoint.name for checkpoint in runs[0][2]],
            ["checkpoint-480"],
        )

    def test_discover_training_runs_without_filter_preserves_numeric_order(self):
        with TemporaryDirectory() as tmp:
            self._create_training_run(tmp)

            runs = discover_training_runs(Path(tmp))

        self.assertEqual(
            [checkpoint.name for checkpoint in runs[0][2]],
            ["checkpoint-0", "checkpoint-160", "checkpoint-480"],
        )

    def test_discover_training_runs_rejects_missing_requested_checkpoint(self):
        with TemporaryDirectory() as tmp:
            self._create_training_run(tmp)

            with self.assertRaisesRegex(ValueError, r"Missing.*\[320\]"):
                discover_training_runs(Path(tmp), checkpoint_steps=[320])


class BuildEvalCmdTests(unittest.TestCase):
    def _saved_config(self):
        return {
            "training_args": {
                "loss_type": "rebel",
                "qrpo_transform_type": "identity",
                "qrpo_loss_type": "mse",
                "qrpo_sample_selector": "both",
                "num_ref_rewards": 6,
                "beta": 1e-4,
            },
            "wandb": {
                "tags": ["model-llama", "dataset-ultrafeedback-armorm"],
            },
            "dataset_args": {
                "dataset_path": "/tmp/magpie-dataset",
                "train_split": {"name": "train"},
                "eval_split": {"name": "test"},
            },
            "model_args": {
                "model_name_or_path": "/tmp/mistral",
            },
            "job_subdir": "chat-baseline/example",
            "data_dir": "/tmp/data",
            "ref_model_key": "mistral",
        }

    def test_build_eval_cmd_preserves_saved_split_names(self):
        saved_config = self._saved_config()

        cmd = build_eval_cmd(
            saved_config=saved_config,
            training_dir="/tmp/training",
            prefix="skywork_v2_qwen_",
            eval_type="eval_split",
            model_name_override="mistral",
            dataset_name_override="magpieair-armorm",
            ref_model_key_override="mistral-sft",
        )

        self.assertIn("dataset_args.train_split.name=train", cmd)
        self.assertIn("dataset_args.eval_split.name=test", cmd)

    def test_build_eval_cmd_uses_explicit_rescore_overrides(self):
        saved_config = self._saved_config()
        saved_config.pop("ref_model_key")

        cmd = build_eval_cmd(
            saved_config=saved_config,
            training_dir="/tmp/training",
            prefix="skywork_v2_qwen_",
            eval_type="alpaca_eval",
            model_name_override="mistral",
            dataset_name_override="magpieair-armorm",
            ref_model_key_override="mistral-sft",
        )

        self.assertIn("model=mistral", cmd)
        self.assertNotIn("model=llama", cmd)
        self.assertIn("dataset=magpieair-armorm", cmd)
        self.assertNotIn("dataset=ultrafeedback-armorm", cmd)
        self.assertIn("++ref_model_key=mistral-sft", cmd)

    def test_build_eval_cmd_rejects_missing_explicit_eval_overrides(self):
        with self.assertRaisesRegex(ValueError, "--model"):
            build_eval_cmd(
                saved_config=self._saved_config(),
                training_dir="/tmp/training",
                prefix="skywork_v2_qwen_",
                eval_type="alpaca_eval",
            )

    def test_build_eval_cmd_includes_normalized_checkpoint_steps(self):
        cmd = build_eval_cmd(
            saved_config=self._saved_config(),
            training_dir="/tmp/training",
            prefix="skywork_v2_qwen_",
            eval_type="eval_split",
            model_name_override="mistral",
            dataset_name_override="magpieair-armorm",
            ref_model_key_override="mistral-sft",
            checkpoint_steps=[480, 160],
        )

        self.assertEqual(
            [arg for arg in cmd if arg.startswith("checkpoint_steps=")],
            ["checkpoint_steps=[160,480]"],
        )

    def test_build_eval_cmd_without_checkpoint_steps_omits_override(self):
        cmd = build_eval_cmd(
            saved_config=self._saved_config(),
            training_dir="/tmp/training",
            prefix="skywork_v2_qwen_",
            eval_type="eval_split",
            model_name_override="mistral",
            dataset_name_override="magpieair-armorm",
            ref_model_key_override="mistral-sft",
        )

        self.assertFalse(
            any(arg.startswith("checkpoint_steps=") for arg in cmd)
        )


if __name__ == "__main__":
    unittest.main()
