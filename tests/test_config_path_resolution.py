import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qrpo.utils.config import resolve_existing_local_path


class ExistingLocalPathResolutionTests(unittest.TestCase):
    def test_resolves_relative_local_path_against_original_repo_root(self):
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            model_dir = repo_root / "data/shared/models/mistral-sft"
            model_dir.mkdir(parents=True)
            checkpoint_dir = repo_root / "outputs/shared/train_qrpo/run/checkpoints/hash"
            checkpoint_dir.mkdir(parents=True)

            old_cwd = Path.cwd()
            try:
                os.chdir(checkpoint_dir)

                resolved = resolve_existing_local_path(
                    "data/shared/models/mistral-sft", repo_root
                )
            finally:
                os.chdir(old_cwd)

            self.assertEqual(resolved, str(model_dir.resolve()))

    def test_leaves_hub_repo_id_unchanged_when_no_local_path_exists(self):
        with TemporaryDirectory() as tmpdir:
            resolved = resolve_existing_local_path(
                "mistralai/Mistral-7B-Instruct-v0.2", Path(tmpdir)
            )

            self.assertEqual(resolved, "mistralai/Mistral-7B-Instruct-v0.2")
