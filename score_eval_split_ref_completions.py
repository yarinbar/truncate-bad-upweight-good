#!/usr/bin/env python3
"""Score eval-split reference completions with a reward model."""

import argparse
import json
import logging
import shutil
from pathlib import Path

import datasets

from generate_alpaca_ref_completions import score_ref_completions

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Score eval-split ref completions with a reward model."
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to merged dataset containing ref_completions on the eval split.",
    )
    parser.add_argument(
        "--split",
        default="test_prefs",
        help="Dataset split to score (default: test_prefs)",
    )
    parser.add_argument(
        "--rm_name",
        required=True,
        help="Short name for the reward model (e.g. skywork_v2_llama)",
    )
    parser.add_argument(
        "--rm_path",
        required=True,
        help="HuggingFace model name/path for the reward model",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing scored dataset",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_path = dataset_path / "eval-split-rewards" / args.rm_name

    if output_path.exists() and not args.overwrite:
        logger.info(f"Already exists: {output_path}. Use --overwrite to rescore.")
        return

    ds = datasets.load_from_disk(str(dataset_path))[args.split]
    logger.info(f"Loaded {len(ds)} rows from {dataset_path}[{args.split}]")

    prompts = []
    completions_per_prompt = []
    for row in ds:
        ref_completions = row["ref_completions"]
        prompts.append(ref_completions[0]["content"])
        completions_per_prompt.append(json.loads(ref_completions[1]["content"]))

    if not prompts:
        raise ValueError(f"No rows found in {dataset_path}[{args.split}]")

    logger.info(
        f"Parsed {len(prompts)} prompts, "
        f"{len(completions_per_prompt[0])} completions each"
    )

    ref_rewards, ref_tokens_len = score_ref_completions(
        prompts,
        completions_per_prompt,
        args.rm_path,
        args.batch_size,
        args.max_seq_length,
        args.num_gpus,
    )

    out_ds = datasets.Dataset.from_dict(
        {
            "ref_rewards": ref_rewards,
            "ref_completions_reward_tokens_len": ref_tokens_len,
        }
    )

    if output_path.exists():
        shutil.rmtree(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_ds.save_to_disk(str(output_path))
    logger.info(f"Saved {len(out_ds)} rows to {output_path}")


if __name__ == "__main__":
    main()
