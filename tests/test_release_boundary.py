import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_TEXT = [
    r"/home/yarinbar", r"/mnt/public", r"nid0\d+", r"clariden", r"capstor",
    r"ssl_ratio", r"ssl_history", r"mix_ours_ssl_targets",
]
FORBIDDEN_FILES = [
    "llama_magpie_ssl.sh", "magpie_ours_hparam_search.sh", "ultrafeedback_ours_hparam_search.sh",
    "magpie_bonbon_hparam_search.sh", "truncation_ablation.sh", "estimate_ours_beta_z.py",
]


def tracked_files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    return [ROOT / p for p in out.stdout.splitlines()]


def test_no_private_paths_hostnames_or_ssl_symbols():
    offenders = []
    for path in tracked_files():
        if path.suffix in {".npz", ".png", ".svg"} or path.name == "test_release_boundary.py":
            continue
        text = path.read_text(errors="ignore")
        for pattern in FORBIDDEN_TEXT:
            if re.search(pattern, text):
                offenders.append(f"{path.relative_to(ROOT)}: {pattern}")
    assert offenders == []


def test_no_search_grid_or_ssl_launchers():
    names = {p.name for p in tracked_files()}
    assert names.isdisjoint(FORBIDDEN_FILES)


def test_wandb_mode_defaults_to_disabled_and_honours_env():
    text = (ROOT / "src/qrpo/configs/setup.yaml").read_text()
    assert "mode: ${oc.env:WANDB_MODE,disabled}" in text
