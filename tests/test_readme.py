from pathlib import Path

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")


def test_readme_covers_reader_workflow():
    for needle in ["bash install.sh", "bash scripts/download_models.sh", "bash scripts/download_data.sh",
                   "bash scripts/score_reference_pools.sh", "bash experiments/llama_8b.sh",
                   "bash experiments/mistral_7b.sh", "scripts/judge_alpaca_gpt4o.sh",
                   "notebooks/static_experiments.ipynb", "loss_type=ours",
                   "TUP_DATA_DIR", "HF_HUB_OFFLINE=1", "scripts/export_static_data.py",
                   "export HF_HUB_OFFLINE=1", "run_rescore_multi_rm.sh", "score_reference_pools.sh --use-hosted",
                   "several hours on 4 GPUs", "tinycss2", "A trained run evaluates and rescores only its reported checkpoint"]:
        assert needle in README, needle
    assert "base-model-evals" not in README


def test_readme_has_no_paper_comparison_sections():
    for gone in ["Figure discrepancies", "Figures: corrected", "arXiv v1", "Known remaining issues",
                 "Deviations from the paper"]:
        assert gone not in README, gone


def test_readme_maps_table_columns_to_log_fields():
    for needle in ["reporting/lc_lcmodel_avg_reward_mean", "avg_reward_tokens_len_mean",
                   "length_controlled_winrate", "lc_standard_error", "checkpoint-0",
                   "initial-<dataset_id>/checkpoints/base/checkpoint-0"]:
        assert needle in README, needle
