import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHA = re.compile(r"^[0-9a-f]{40}$")


def dry(script, *args, data_dir="/tmp/tup-data", prefix="+ hf download "):
    res = subprocess.run(["bash", str(ROOT / "scripts" / script), *args, "--dry-run"], cwd=ROOT,
                         capture_output=True, text=True, env={**os.environ, "TUP_DATA_DIR": data_dir})
    assert res.returncode == 0, res.stderr
    return [l for l in res.stdout.splitlines() if l.startswith(prefix)]


def pins():
    script = f'source "{ROOT}/scripts/pins.sh"; for v in ${{!PIN_@}}; do echo "${{!v}}"; done'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.splitlines()


def test_every_third_party_pin_is_a_full_sha():
    entries = pins()
    assert len(entries) == 15
    for entry in entries:
        repo, rev, _type, _dest = entry.split()
        assert SHA.match(rev) or (repo.startswith("yarinbar/") and rev == "UNPUBLISHED"), entry


def test_download_models_default_fetches_policy_and_reward_models():
    lines = dry("download_models.sh")
    joined = "\n".join(lines)
    for repo in ["allenai/Llama-3.1-Tulu-3-8B-SFT", "RLHFlow/ArmoRM-Llama3-8B-v0.1",
                 "Skywork/Skywork-Reward-V2-Llama-3.1-8B", "Skywork/Skywork-Reward-V2-Qwen3-8B",
                 "yarinbar/tup-mistral-7b-sft-magpieair"]:
        assert repo in joined
    assert "mistralai/Mistral-7B-Instruct-v0.2" not in joined
    assert all("--revision " in l for l in lines)


def test_download_models_sft_base_alone_fetches_only_the_sft_start_model():
    lines = dry("download_models.sh", "--sft-base")
    assert len(lines) == 1 and "mistralai/Mistral-7B-Instruct-v0.2" in lines[0]
    assert dry("download_models.sh", "--sft-base", prefix="+ pin refs/main ") == []


def test_download_models_policy_flags_bring_the_reward_models():
    for flag, policy in (("--llama", "allenai/Llama-3.1-Tulu-3-8B-SFT"),
                         ("--mistral", "yarinbar/tup-mistral-7b-sft-magpieair")):
        joined = "\n".join(dry("download_models.sh", flag, "--sft-base"))
        for repo in [policy, "mistralai/Mistral-7B-Instruct-v0.2", "RLHFlow/ArmoRM-Llama3-8B-v0.1",
                     "Skywork/Skywork-Reward-V2-Llama-3.1-8B", "Skywork/Skywork-Reward-V2-Qwen3-8B"]:
            assert repo in joined, (flag, repo)


def test_download_data_tup_only_skips_best_pools():
    lines = dry("download_data.sh", "--setting", "all", "--tup-only")
    joined = "\n".join(lines)
    assert "offpolicy2random" in joined and "offpolicy2best" not in joined
    assert "tatsu-lab/alpaca_eval" in joined and "yarinbar/tup-eval-artifacts" in joined


def test_download_data_sft_fetches_the_exact_sft_set():
    joined = "\n".join(dry("download_data.sh", "--setting", "mistral-magpie", "--sft"))
    assert "qrpo-paper-mistral-nosft-magpieair-armorm-temp1-ref50-offpolicy2best-armorm" in joined
    assert "9333194668c68782171ffa57b7c47cb18f17bec0" in joined


SKYWORK_PINS = [("Skywork/Skywork-Reward-V2-Llama-3.1-8B", "cba2f842f3f1af2f1b2f0d35e794d789976390c5"),
                ("Skywork/Skywork-Reward-V2-Qwen3-8B", "6f19fdefb933293d4898bdb59a96f7223d998659")]


def test_download_models_dry_run_announces_main_ref_pins_for_cache_models():
    lines = dry("download_models.sh", prefix="+ pin refs/main ")
    assert lines == [f"+ pin refs/main {repo} {sha}" for repo, sha in SKYWORK_PINS]


def test_download_models_points_cache_main_ref_at_the_pinned_sha(tmp_path):
    # The launchers look the Skywork models up by repo id without a revision, which needs refs/main;
    # `hf download <repo> --revision <sha>` writes no ref, so the script must.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_hf = fake_bin / "hf"
    fake_hf.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_hf.chmod(0o755)
    path = os.pathsep.join([str(fake_bin), str(Path(sys.executable).parent), os.environ["PATH"]])
    env = {**os.environ, "PATH": path, "HF_HUB_CACHE": str(tmp_path / "hub"), "TUP_DATA_DIR": str(tmp_path / "data")}
    res = subprocess.run(["bash", str(ROOT / "scripts/download_models.sh")], cwd=ROOT,
                         capture_output=True, text=True, env=env)
    assert res.returncode == 1  # stops at the not-yet-published mistral-sft entry, after the Skywork ones
    assert "yarinbar/tup-mistral-7b-sft-magpieair is not published yet" in res.stderr
    for repo, sha in SKYWORK_PINS:
        ref = tmp_path / "hub" / f"models--{repo.replace('/', '--')}" / "refs" / "main"
        assert ref.read_text() == sha  # exact SHA, no trailing newline (as huggingface_hub writes refs)
