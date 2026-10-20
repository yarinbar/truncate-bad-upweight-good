import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETS = [
    ("llama-nosft-ultrafeedback-armorm-temp1-ref50-offpolicy2random-armorm", "test_prefs"),
    ("llama-nosft-magpieair-armorm-temp1-ref50-offpolicy2best-armorm", "test"),
    ("mistral-sft-magpieair-armorm-temp1-ref50-offpolicy2random-armorm", "test"),
]


def test_dry_run_scores_each_present_set_with_both_skywork_models(tmp_path):
    merged = tmp_path / "shared/datasets-with-ref-rewards/merged"
    for name, _ in SETS:
        (merged / name).mkdir(parents=True)
    res = subprocess.run(["bash", str(ROOT / "scripts/score_reference_pools.sh"), "--dry-run"], cwd=ROOT,
                         capture_output=True, text=True, env={**os.environ, "TUP_DATA_DIR": str(tmp_path)})
    assert res.returncode == 0, res.stderr
    lines = [l for l in res.stdout.splitlines() if l.startswith("+ python")]
    for name, split in SETS:
        for rm, path in [("skywork_v2_llama", "Skywork/Skywork-Reward-V2-Llama-3.1-8B"),
                         ("skywork_v2_qwen", "Skywork/Skywork-Reward-V2-Qwen3-8B")]:
            assert any(f"merged/{name}" in l and f"--split {split}" in l and f"--rm_name {rm}" in l
                       and f"--rm_path {path}" in l for l in lines), (name, rm)
    for model in ["llama", "mistral-sft"]:
        for rm in ["armorm", "skywork_v2_llama", "skywork_v2_qwen"]:
            assert any("generate_alpaca_ref_completions.py" in l and "--skip_generation" in l
                       and f"--model_key {model}" in l and f"--rm_name {rm}" in l for l in lines), (model, rm)


def test_script_runs_offline_for_hf_cache(tmp_path):
    text = (ROOT / "scripts/score_reference_pools.sh").read_text()
    assert "export HF_HUB_OFFLINE=1" in text


def test_use_hosted_dry_run_links_artifacts_and_creates_parent_dirs(tmp_path):
    merged = tmp_path / "shared/datasets-with-ref-rewards/merged"
    for name, _ in SETS:
        (merged / name).mkdir(parents=True)
    res = subprocess.run(["bash", str(ROOT / "scripts/score_reference_pools.sh"), "--use-hosted", "--dry-run"],
                         cwd=ROOT, capture_output=True, text=True, env={**os.environ, "TUP_DATA_DIR": str(tmp_path)})
    assert res.returncode == 0, res.stderr
    lines = [l for l in res.stdout.splitlines() if l.startswith("+")]
    assert any("ln -sfn" in l and "tup-eval-artifacts/eval-split-rewards/" in l for l in lines)
    assert any("mkdir -p" in l and "eval-split-rewards" in l for l in lines)
    assert any("ln -sfn" in l and "tup-eval-artifacts/alpaca-ref-completions/llama/armorm" in l for l in lines)
    assert any("mkdir -p" in l and "alpaca-ref-completions/mistral-sft" in l for l in lines)
    assert not any(l.startswith("+ python") for l in lines)


SET_NAME = "llama-nosft-magpieair-armorm-temp1-ref50-offpolicy2best-armorm"
ALPACA_RMS = ["armorm", "skywork_v2_llama", "skywork_v2_qwen"]


def _real_hosted(data_dir):
    return subprocess.run(["bash", str(ROOT / "scripts/score_reference_pools.sh"), "--use-hosted"], cwd=ROOT,
                          capture_output=True, text=True, env={**os.environ, "TUP_DATA_DIR": str(data_dir)})


def _make_merged(data_dir):
    (data_dir / "shared/datasets-with-ref-rewards/merged" / SET_NAME).mkdir(parents=True)


def _make_artifacts(data_dir):
    art = data_dir / "shared/tup-eval-artifacts"
    for rm in ["skywork_v2_llama", "skywork_v2_qwen"]:
        (art / "eval-split-rewards" / SET_NAME / rm).mkdir(parents=True)
    for model in ["llama", "mistral-sft"]:
        for rm in ALPACA_RMS:
            (art / "alpaca-ref-completions" / model / rm).mkdir(parents=True)


def test_hosted_real_run_fails_without_artifacts_and_creates_no_dangling_link(tmp_path):
    _make_merged(tmp_path)
    res = _real_hosted(tmp_path)
    assert res.returncode != 0
    assert "missing hosted scores" in res.stderr
    dest = tmp_path / "shared/datasets-with-ref-rewards/merged" / SET_NAME / "eval-split-rewards/skywork_v2_llama"
    assert not os.path.lexists(dest)


def test_hosted_real_run_links_resolve_when_artifacts_exist(tmp_path):
    _make_merged(tmp_path)
    _make_artifacts(tmp_path)
    res = _real_hosted(tmp_path)
    assert res.returncode == 0, res.stderr
    ds = tmp_path / "shared/datasets-with-ref-rewards/merged" / SET_NAME
    for rm in ["skywork_v2_llama", "skywork_v2_qwen"]:
        assert os.path.isdir(ds / "eval-split-rewards" / rm)
    for model in ["llama", "mistral-sft"]:
        for rm in ALPACA_RMS:
            assert os.path.isdir(tmp_path / "shared/datasets/alpaca-ref-completions" / model / rm)


def test_hosted_real_run_refuses_to_nest_over_local_directory(tmp_path):
    _make_merged(tmp_path)
    _make_artifacts(tmp_path)
    local = tmp_path / "shared/datasets-with-ref-rewards/merged" / SET_NAME / "eval-split-rewards/skywork_v2_llama"
    local.mkdir(parents=True)
    res = _real_hosted(tmp_path)
    assert res.returncode != 0
    assert "already exists" in res.stderr
    assert not os.path.islink(local)
    assert not any(local.iterdir())
