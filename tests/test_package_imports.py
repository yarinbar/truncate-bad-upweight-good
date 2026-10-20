import importlib

import pytest

MODULES = [
    "qrpo.train_qrpo",
    "qrpo.train_sft",
    "qrpo.train_stop_callback",
    "qrpo.trainers.qrpo",
    "qrpo.trainers.qrpo_config",
    "qrpo.evals.run_online_eval",
    "qrpo.evals.create_eval_split",
    "qrpo.evals.checkpoint_filter",
    "qrpo.evals.run_alpaca_gpt",
    "qrpo.evals.judge_metadata",
]


@pytest.mark.parametrize("name", MODULES)
def test_release_module_imports(name):
    importlib.import_module(name)
