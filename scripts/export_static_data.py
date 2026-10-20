"""Export the reward arrays used by notebooks/static_experiments.ipynb (no GPU, no model).

Needs the Llama UltraFeedback and Magpie Air offpolicy2random QRPO sets (scripts/download_data.sh) with their
Skywork eval-split scores (scripts/score_reference_pools.sh). Writes notebooks/static_data.npz.
"""
import argparse
from pathlib import Path

import numpy as np

from static_data import load_static_reward_arrays  # scripts/static_data.py (verbatim from the paper code)

# Exact data selection of every panel (found by matching the regenerated SVGs; spec §8).
# One entry per figure file: the eval split of the offpolicy2random set, scored by the one Skywork RM it names.
PANELS = [
    {"figure": "fig2", "plot": "region", "dataset": "ultrafeedback", "variant": "random", "source": "merged",
     "eval_rms": ["skywork_v2_llama"], "file_label": "Skywork-Llama"},
    {"figure": "fig2", "plot": "benefit", "dataset": "ultrafeedback", "variant": "random", "source": "merged",
     "eval_rms": ["skywork_v2_llama"], "file_label": "Skywork-Llama"},
    {"figure": "fig6", "plot": "region", "dataset": "ultrafeedback", "variant": "random", "source": "merged",
     "eval_rms": ["skywork_v2_qwen"], "file_label": "Skywork-Qwen"},
    {"figure": "fig6", "plot": "region", "dataset": "magpie", "variant": "random", "source": "merged",
     "eval_rms": ["skywork_v2_llama"], "file_label": "Skywork-Llama"},
    {"figure": "fig6", "plot": "region", "dataset": "magpie", "variant": "random", "source": "merged",
     "eval_rms": ["skywork_v2_qwen"], "file_label": "Skywork-Qwen"},
    {"figure": "fig7", "plot": "benefit", "dataset": "ultrafeedback", "variant": "random", "source": "merged",
     "eval_rms": ["skywork_v2_qwen"], "file_label": "Skywork-Qwen"},
    {"figure": "fig7", "plot": "benefit", "dataset": "magpie", "variant": "random", "source": "merged",
     "eval_rms": ["skywork_v2_llama"], "file_label": "Skywork-Llama"},
    {"figure": "fig7", "plot": "benefit", "dataset": "magpie", "variant": "random", "source": "merged",
     "eval_rms": ["skywork_v2_qwen"], "file_label": "Skywork-Qwen"},
]
# App. B.3: Llama UF, random variant, merged eval split, ArmoRM proxy vs Skywork-Llama oracle (rebuttal.ipynb cell 21).
B3 = {"dataset": "ultrafeedback", "variant": "random", "source": "merged", "eval_rms": ["skywork_v2_llama"]}


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=repo / "data", help="root containing shared/ (default: repo data/)")
    parser.add_argument("--out", type=Path, default=repo / "notebooks/static_data.npz")
    args = parser.parse_args()
    shared = args.data_dir / "shared"
    arrays = {}
    for sel in [*PANELS, B3]:
        rewards, _ = load_static_reward_arrays("llama", sel["dataset"], eval_rms=sel["eval_rms"],
                                               variant=sel["variant"], eval_source=sel["source"],
                                               base_dir=shared / "datasets-with-ref-rewards/merged",
                                               alpaca_dir=shared / "datasets/alpaca-ref-completions")
        for rm, values in rewards.items():
            arrays[f'{sel["dataset"]}/{sel["variant"]}/{sel["source"]}/{rm}'] = np.asarray(values, dtype=np.float64)
    np.savez_compressed(args.out, **arrays)
    print(f"wrote {len(arrays)} arrays to {args.out}")


if __name__ == "__main__":
    main()
