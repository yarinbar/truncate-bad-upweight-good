import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Optional

import hydra
import pandas as pd
import wandb
from omegaconf import DictConfig

from qrpo import utils
from qrpo.evals.judge_metadata import (
    ensure_judge_config,
    infer_judge_model,
    resolve_local_judge_config,
)

utils.config.register_resolvers()
logger = logging.getLogger(__name__)
DEFAULT_JUDGE_CONFIG = "alpaca_eval_gpt4_turbo_fn"


def get_judge_configs(config: DictConfig) -> list[str]:
    judge_configs = list(getattr(config, "annotators_configs", []) or [])
    if judge_configs:
        return judge_configs
    legacy_judge = getattr(config, "annotators_config", None)
    if legacy_judge:
        return [legacy_judge]
    return [DEFAULT_JUDGE_CONFIG]


def get_wandb_run_id_and_global_step(checkpoint_path: Path) -> tuple[str, int]:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.name, 0
    return checkpoint_path.parent.name, int(checkpoint_path.name.split("-")[-1])


def build_alpaca_eval_cmd(
    completions_file: Path, checkpoint_path: Path, judge_config: str
) -> list[str]:
    resolved_judge_config = ensure_judge_config(judge_config)
    output_path = get_alpaca_eval_output_path(checkpoint_path, judge_config)
    return [
        "alpaca_eval",
        "--model_outputs",
        str(completions_file),
        "--annotators_config",
        resolved_judge_config,
        "--output_path",
        str(output_path.absolute()),
    ]


def get_alpaca_eval_output_path(checkpoint_path: Path, judge_config: str) -> Path:
    if resolve_local_judge_config(judge_config) is not None:
        return checkpoint_path / judge_config
    return checkpoint_path


def _has_root_level_judge_results(checkpoint_path: Path, judge_config: str) -> bool:
    leaderboard_file = checkpoint_path / "leaderboard.csv"
    annotations_file = checkpoint_path / "annotations.json"
    if not leaderboard_file.exists() or not annotations_file.exists():
        return False

    try:
        with open(annotations_file, "r") as f:
            annotations = json.load(f)
    except (json.JSONDecodeError, OSError, TypeError):
        return False

    annotators = {
        str(annotation.get("annotator")).strip()
        for annotation in annotations
        if isinstance(annotation, dict) and annotation.get("annotator")
    }
    return annotators == {judge_config}


def find_judge_results_dir(checkpoint_path: Path, judge_config: str) -> Optional[Path]:
    judge_dir = checkpoint_path / judge_config
    if (judge_dir / "leaderboard.csv").exists() or (judge_dir / "annotations.json").exists():
        return judge_dir
    if _has_root_level_judge_results(checkpoint_path, judge_config):
        return checkpoint_path
    return None


def validate_annotations(
    checkpoint_path: Path, judge_config: str, expected_count: int
) -> None:
    results_dir = find_judge_results_dir(checkpoint_path, judge_config)
    if results_dir is None:
        return

    annotations_file = results_dir / "annotations.json"
    if not annotations_file.exists():
        return

    with open(annotations_file, "r") as f:
        annotations = json.load(f)

    if len(annotations) != expected_count:
        raise RuntimeError(
            f"{judge_config}: expected {expected_count} annotations, found {len(annotations)}"
        )

    def has_nonempty_outputs(annotation: dict[str, object]) -> bool:
        if "output_1" in annotation or "output_2" in annotation:
            return all(str(annotation.get(key, "")).strip() for key in ("output_1", "output_2"))
        return bool(str(annotation.get("output", "")).strip())

    empty_outputs = [a for a in annotations if not has_nonempty_outputs(a)]
    missing_preferences = [a for a in annotations if a.get("preference") is None]
    if empty_outputs or missing_preferences:
        raise RuntimeError(
            f"{judge_config}: invalid annotations "
            f"(empty_outputs={len(empty_outputs)}, missing_preferences={len(missing_preferences)})"
        )


def maybe_run_judge_eval(
    checkpoint_path: Path, completions_file: Path, judge_config: str
) -> None:
    existing_results_dir = find_judge_results_dir(checkpoint_path, judge_config)
    if existing_results_dir is not None:
        logger.info(
            "Skipping Alpaca Eval 2 for %s: results already exist at %s",
            judge_config,
            existing_results_dir,
        )
        return

    logger.info("Running Alpaca Eval 2 for %s...", judge_config)
    cmd_alpaca_eval = build_alpaca_eval_cmd(completions_file, checkpoint_path, judge_config)
    logger.info("Running command: %s", " ".join(cmd_alpaca_eval))
    subprocess.run(cmd_alpaca_eval, check=True)


def _coerce_scalar(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value


def _make_legacy_logs(
    summary_row: dict[str, object],
    judge_config: str,
    judge_model: str,
    global_step: int,
) -> dict[str, object]:
    logs = {f"alpaca/{k}": str(v) for k, v in summary_row.items()}
    logs["train/global_step"] = global_step
    logs["judge_config"] = judge_config
    logs["judge_model"] = judge_model
    return logs


def _make_wandb_logs(
    summary_row: dict[str, object],
    judge_config: str,
    global_step: int,
    legacy_mode: bool,
) -> dict[str, object]:
    prefix = "alpaca" if legacy_mode else f"alpaca/{judge_config}"
    logs = {f"{prefix}/{k}": _coerce_scalar(v) for k, v in summary_row.items()}
    logs["train/global_step"] = global_step
    return logs


def write_judge_summaries(
    checkpoint_path: Path, judge_configs: list[str], global_step: int
) -> pd.DataFrame:
    summary_rows = []
    legacy_only = judge_configs == [DEFAULT_JUDGE_CONFIG]

    for judge_config in judge_configs:
        results_dir = find_judge_results_dir(checkpoint_path, judge_config)
        csv_file = (results_dir / "leaderboard.csv") if results_dir is not None else None
        if csv_file is None or not csv_file.exists():
            expected_csv_file = checkpoint_path / judge_config / "leaderboard.csv"
            raise FileNotFoundError(
                f"Alpaca Eval results for {judge_config} not found at {expected_csv_file}"
            )

        alpaca_results = pd.read_csv(csv_file)
        summary_row = dict(alpaca_results.iloc[0])
        judge_model = infer_judge_model(judge_config)
        summary_rows.append(
            {"judge_config": judge_config, "judge_model": judge_model, **summary_row}
        )

        logs = _make_legacy_logs(summary_row, judge_config, judge_model, global_step)
        logs_path = checkpoint_path / f"{judge_config}.json"
        with open(logs_path, "w") as f:
            json.dump(logs, f)

        if legacy_only:
            legacy_logs_path = checkpoint_path / f"{DEFAULT_JUDGE_CONFIG}.json"
            if legacy_logs_path != logs_path:
                with open(legacy_logs_path, "w") as f:
                    json.dump(logs, f)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(checkpoint_path / "alpaca_eval_leaderboard_by_judge.csv", index=False)
    return summary_df


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="run-alpaca-gpt",
)
def main(config: DictConfig) -> None:
    if "OPENAI_API_KEY" not in os.environ:
        logger.error(
            "Error: OPENAI_API_KEY is not set. Please set it before running the script."
        )
        sys.exit(1)

    random.seed(config.seed)
    judge_configs = get_judge_configs(config)

    # Ensure output directory exists
    checkpoint_path = Path(config.checkpoint_path)
    completions_file = checkpoint_path / "online_evals_0/alpaca_eval/model_outputs.json"

    # Check if output file already exists
    if not completions_file.exists():
        logger.error(f"Error: Completions file not found at {completions_file}")
        sys.exit(1)

    logger.info(
        "Start evaluation for judges: %s",
        ", ".join(
            f"{judge_config} (judge_model={infer_judge_model(judge_config)})"
            for judge_config in judge_configs
        ),
    )
    with open(completions_file, "r") as f:
        completions = json.load(f)
    expected_annotations = len(completions)

    for judge_config in judge_configs:
        maybe_run_judge_eval(checkpoint_path, completions_file, judge_config)
        validate_annotations(checkpoint_path, judge_config, expected_annotations)

    logger.info("Alpaca Eval queries completed. Results are in %s", checkpoint_path)

    wandb_run_id, global_step = get_wandb_run_id_and_global_step(checkpoint_path)
    wandb.init(
        id=wandb_run_id,
        resume="allow",
        mode=config.wandb.mode,
        project=config.wandb.project,
    )

    summary_df = write_judge_summaries(checkpoint_path, judge_configs, global_step)
    legacy_only = judge_configs == [DEFAULT_JUDGE_CONFIG]
    for judge_config in judge_configs:
        judge_df = summary_df[summary_df["judge_config"] == judge_config].drop(
            columns=["judge_config"]
        )
        summary_row = dict(judge_df.iloc[0])
        wandb_logs = _make_wandb_logs(summary_row, judge_config, global_step, legacy_only)
        table_key = (
            f"alpaca/evals_at_{global_step}"
            if legacy_only
            else f"alpaca/{judge_config}/evals_at_{global_step}"
        )
        wandb.log({table_key: wandb.Table(dataframe=judge_df), **wandb_logs})

    logger.info("Alpaca Eval pipeline completed. Results are in %s", checkpoint_path)


if __name__ == "__main__":
    main()
