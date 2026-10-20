import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def test_indices_follow_create_eval_split_logic(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts/make_eval_split_indices.py"), "--data-dir", str(tmp_path)],
                   check=True)
    for name, n in [("ultrafeedback-armorm", 1995), ("magpieair-armorm", 1992)]:
        d = tmp_path / "shared/datasets-with-chosen-rewards/merged" / name
        reporting, selection = train_test_split(np.arange(n), test_size=0.5, random_state=43)
        assert np.array_equal(np.load(d / "eval-reward-reporting-indices.npy"), reporting)
        assert np.array_equal(np.load(d / "eval-checkpoint-selection-indices.npy"), selection)
    assert len(np.load(tmp_path / "shared/datasets-with-chosen-rewards/merged/ultrafeedback-armorm/eval-checkpoint-selection-indices.npy")) == 998
