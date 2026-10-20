"""Write the checkpoint-selection / reward-reporting index files used by run_online_eval.py.

Same logic as qrpo.evals.create_eval_split (50/50 split, random_state=43), computed from the
eval-split row counts so the 45 GB chosen-rewards datasets are not needed.
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

EVAL_ROWS = {"ultrafeedback-armorm": 1995, "magpieair-armorm": 1992}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    for name, n_rows in EVAL_ROWS.items():
        out = args.data_dir / "shared/datasets-with-chosen-rewards/merged" / name
        out.mkdir(parents=True, exist_ok=True)
        reporting, selection = train_test_split(np.arange(n_rows), test_size=0.5, random_state=43)
        np.save(out / "eval-reward-reporting-indices.npy", reporting)
        np.save(out / "eval-checkpoint-selection-indices.npy", selection)
        print(f"{name}: reporting={len(reporting)} selection={len(selection)} -> {out}")


if __name__ == "__main__":
    main()
