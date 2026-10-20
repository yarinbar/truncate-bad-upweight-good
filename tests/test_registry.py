import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS = ["dpo", "rebel", "rebel-random", "qrpo", "qrpo-random", "bonbon", "tup-mild", "tup-mid", "tup-aggressive"]
UF, MA, MS = "llama-nosft-ultrafeedback", "llama-nosft-magpieair", "mistral-sft-magpieair"

# (setting, run): (dataset prefix, pool, loss, beta, lr, refs, lambda, selector, horizon, stop) — spec §7.2
EXPECTED = {
    ("uf", "dpo"): (UF, "best", "dpo", "0.01", "3e-7", "6", None, "both", "476", "300"),
    ("uf", "rebel"): (UF, "best", "rebel", "1e-6", "3e-7", "6", None, "both", "476", "200"),
    ("uf", "rebel-random"): (UF, "random", "rebel", "1e-6", "1e-6", "6", None, "random", "476", "100"),
    ("uf", "qrpo"): (UF, "best", "qrpo", "3e-4", "3e-7", "3", None, "both", "476", "100"),
    ("uf", "qrpo-random"): (UF, "random", "qrpo", "3e-4", "3e-7", "3", None, "random", "476", "100"),
    ("uf", "bonbon"): (UF, "best", "bonbon", "1e-3", "1e-6", "6", "0.5", "both", "476", "476"),
    ("uf", "tup-mild"): (UF, "random", "ours", "0.01", "1e-7", "6", "0.2", "random", "476", "400"),
    ("uf", "tup-mid"): (UF, "random", "ours", "0.01", "1e-7", "6", "0.5", "random", "476", "400"),
    ("uf", "tup-aggressive"): (UF, "random", "ours", "0.01", "1e-7", "6", "0.8", "random", "476", "476"),
    ("magpie", "dpo"): (MA, "random", "dpo", "0.01", "1e-6", "6", None, "both", "480", "480"),
    ("magpie", "rebel"): (MA, "best", "rebel", "1e-6", "1e-7", "6", None, "both", "640", "640"),
    ("magpie", "rebel-random"): (MA, "random", "rebel", "1e-4", "1e-6", "6", None, "both", "320", "320"),
    ("magpie", "qrpo"): (MA, "best", "qrpo", "1e-3", "1e-7", "1", None, "both", "320", "320"),
    ("magpie", "qrpo-random"): (MA, "random", "qrpo", "1e-3", "1e-7", "1", None, "both", "320", "320"),
    ("magpie", "bonbon"): (MA, "best", "bonbon", "1e-3", "1e-6", "6", "0.5", "both", "764", "764"),
    ("magpie", "tup-mild"): (MA, "random", "ours", "0.01", "1e-7", "6", "0.2", "random", "764", "320"),
    ("magpie", "tup-mid"): (MA, "random", "ours", "0.01", "1e-7", "6", "0.5", "random", "764", "320"),
    ("magpie", "tup-aggressive"): (MA, "random", "ours", "0.01", "1e-7", "6", "0.8", "random", "764", "320"),
    ("mistral", "dpo"): (MS, "random", "dpo", "0.03", "3e-7", "6", None, "both", "764", "480"),
    ("mistral", "rebel"): (MS, "best", "rebel", "1e-4", "1e-7", "6", None, "both", "764", "480"),
    ("mistral", "rebel-random"): (MS, "random", "rebel", "1e-4", "3e-7", "6", None, "both", "764", "320"),
    ("mistral", "qrpo"): (MS, "best", "qrpo", "3e-3", "1e-7", "1", None, "both", "764", "764"),
    ("mistral", "qrpo-random"): (MS, "random", "qrpo", "3e-3", "1e-6", "1", None, "both", "764", "764"),
    ("mistral", "bonbon"): (MS, "best", "bonbon", "1e-3", "1e-7", "6", "0.5", "both", "764", "480"),
    ("mistral", "tup-mild"): (MS, "random", "ours", "0.01", "1e-6", "6", "0.2", "random", "764", "764"),
    ("mistral", "tup-mid"): (MS, "random", "ours", "0.01", "1e-6", "6", "0.5", "random", "764", "764"),
    ("mistral", "tup-aggressive"): (MS, "random", "ours", "0.01", "1e-6", "6", "0.8", "random", "764", "764"),
}


def registry(setting, run, *extra, env=None):
    cmd = ["bash", str(ROOT / "experiments/mistral_7b.sh")] if setting == "mistral" else \
          ["bash", str(ROOT / "experiments/llama_8b.sh"), setting]
    return subprocess.run([*cmd, run, *extra], cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, **(env or {})})


def overrides(stdout):
    line = next(l for l in stdout.splitlines() if l.startswith("+ accelerate launch"))
    toks = shlex.split(line[2:])
    return toks, dict(t.split("=", 1) for t in toks[toks.index("qrpo.train_qrpo") + 1:] if "=" in t)


UF_RANDOM_ID = "llama-nosft-ultrafeedback-armorm-temp1-ref50-offpolicy2random-armorm"
SKYWORK_PINS = {"Skywork/Skywork-Reward-V2-Llama-3.1-8B": "cba2f842f3f1af2f1b2f0d35e794d789976390c5",
                "Skywork/Skywork-Reward-V2-Qwen3-8B": "6f19fdefb933293d4898bdb59a96f7223d998659"}
# Every input a real `llama_8b.sh uf tup-mid` run checks before launching (relative to the data dir).
UF_TUP_INPUTS = [
    "shared/models/llama", "shared/models/armorm",
    f"shared/datasets-with-ref-rewards/merged/{UF_RANDOM_ID}",
    "shared/datasets-with-chosen-rewards/merged/ultrafeedback-armorm/eval-checkpoint-selection-indices.npy",
    "shared/datasets-with-chosen-rewards/merged/ultrafeedback-armorm/eval-reward-reporting-indices.npy",
    "shared/datasets/alpaca",
    *(f"shared/datasets/alpaca-ref-completions/llama/{d}"
      for d in ("raw-completions", "armorm", "skywork_v2_llama", "skywork_v2_qwen")),
]
# The same for `mistral_7b.sh tup-mid` (Magpie Air split indices, mistral-sft reference pools).
MISTRAL_TUP_INPUTS = [
    "shared/models/mistral-sft", "shared/models/armorm",
    "shared/datasets-with-ref-rewards/merged/mistral-sft-magpieair-armorm-temp1-ref50-offpolicy2random-armorm",
    "shared/datasets-with-chosen-rewards/merged/magpieair-armorm/eval-checkpoint-selection-indices.npy",
    "shared/datasets-with-chosen-rewards/merged/magpieair-armorm/eval-reward-reporting-indices.npy",
    "shared/datasets/alpaca",
    *(f"shared/datasets/alpaca-ref-completions/mistral-sft/{d}"
      for d in ("raw-completions", "armorm", "skywork_v2_llama", "skywork_v2_qwen")),
]


def make_inputs(data, hub, skip=(), inputs=UF_TUP_INPUTS):
    """Create a data tree and HF hub cache holding every checked input of a tup-mid run, except `skip`."""
    for rel in inputs:
        if rel in skip:
            continue
        path = data / rel
        if rel.endswith(".npy"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
        else:
            path.mkdir(parents=True, exist_ok=True)
    for repo, sha in SKYWORK_PINS.items():
        if repo in skip:
            continue
        repo_dir = hub / f"models--{repo.replace('/', '--')}"
        (repo_dir / "snapshots" / sha).mkdir(parents=True)
        (repo_dir / "refs").mkdir()
        (repo_dir / "refs" / "main").write_text(sha)
    return {"TUP_DATA_DIR": str(data), "HF_HUB_CACHE": str(hub)}


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_registry_entry_matches_paper_table(key):
    prefix, pool, loss, beta, lr, refs, lam, sel, horizon, stop = EXPECTED[key]
    res = registry(*key, "--dry-run")
    assert res.returncode == 0, res.stderr
    toks, ov = overrides(res.stdout)
    assert ov["dataset_args.dataset_path"].endswith(f"/{prefix}-armorm-temp1-ref50-offpolicy2{pool}-armorm")
    assert ov["training_args.loss_type"] == loss
    assert float(ov["training_args.beta"]) == float(beta)
    assert float(ov["training_args.learning_rate"]) == float(lr)
    assert ov["training_args.num_ref_rewards"] == refs
    assert ov.get("training_args.binomial_lambda") == lam
    assert ov["training_args.qrpo_sample_selector"] == sel
    assert ov["training_args.max_steps"] == horizon
    assert ov["stop_after_steps"] == stop
    assert toks[toks.index("--num_processes") + 1] == "4"
    if key[0] == "mistral":
        assert ov["model_args.model_name_or_path"].endswith("/data/shared/models/mistral-sft")
        assert ov["++ref_model_key"] == "mistral-sft"
        assert ov["training_args.per_device_eval_batch_size"] == "8"
    else:
        assert ov["model_args.model_name_or_path"].endswith("/data/shared/models/llama")
        assert ov["training_args.per_device_eval_batch_size"] == "4"
    if loss in {"ours", "bonbon"}:
        assert ov["training_args.use_discrete_partition"] == "False"
    # Checkpoint cadence: the launcher passes save_steps; the in-training eval cadence is not passed and
    # comes from the dataset config. Compose the training config to check both effective values.
    cadence = 100 if key[0] == "uf" else 160
    assert ov["training_args.save_steps"] == str(cadence)
    assert "training_args.eval_steps" not in ov
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(ROOT / "src/qrpo/configs"), version_base=None, job_name="t"):
        cfg = compose(config_name="train-qrpo", overrides=toks[toks.index("qrpo.train_qrpo") + 1:])
    assert cfg.training_args.save_steps == cadence
    assert cfg.training_args.eval_steps == cadence


@pytest.mark.parametrize("setting", ["uf", "magpie", "mistral"])
def test_initial_is_base_only_eval(setting):
    res = registry(setting, "initial", "--dry-run")
    assert res.returncode == 0, res.stderr
    assert "+ eval-base-only " in res.stdout


def test_registry_unknown_run_lists_valid_names():
    res = registry("uf", "tup_mid", "--dry-run")
    assert res.returncode != 0
    assert "Unknown run 'tup_mid'" in res.stderr
    assert "tup-mid" in res.stderr and "qrpo-random" in res.stderr


def test_registry_rejects_non_default_nproc():
    res = registry("uf", "tup-mid", "--dry-run", env={"NPROC_PER_NODE": "2"})
    assert res.returncode != 0
    assert "NPROC_PER_NODE must be 4" in res.stderr


def test_registry_missing_inputs_names_download_command(tmp_path):
    env = {"TUP_DATA_DIR": str(tmp_path / "data")}
    res = registry("uf", "tup-mid", env=env)  # not a dry run: input checks run
    assert res.returncode != 0
    assert "scripts/download_data.sh" in res.stderr or "scripts/download_models.sh" in res.stderr
    assert str(tmp_path / "data") in res.stderr


@pytest.mark.parametrize("missing,hint", [
    ("shared/datasets-with-chosen-rewards/merged/ultrafeedback-armorm/eval-reward-reporting-indices.npy",
     "bash scripts/download_data.sh"),
    ("shared/datasets/alpaca", "bash scripts/download_data.sh"),
    ("shared/datasets/alpaca-ref-completions/llama/raw-completions", "bash scripts/download_data.sh"),
    ("shared/datasets/alpaca-ref-completions/llama/armorm", "bash scripts/score_reference_pools.sh"),
    ("shared/datasets/alpaca-ref-completions/llama/skywork_v2_qwen", "bash scripts/score_reference_pools.sh"),
    ("shared/models/armorm", "bash scripts/download_models.sh"),
    ("Skywork/Skywork-Reward-V2-Qwen3-8B", "bash scripts/download_models.sh"),
])
def test_registry_missing_eval_inputs_fail_before_training(tmp_path, missing, hint):
    # Inputs that only the eval stages read must be checked up front, not after hours of training.
    env = make_inputs(tmp_path / "data", tmp_path / "hub", skip={missing})
    marker = tmp_path / "launcher_ran"
    fake = tmp_path / "fake_launcher.sh"
    fake.write_text(f'#!/usr/bin/env bash\ntouch {shlex.quote(str(marker))}\nexit 0\n')
    fake.chmod(0o755)
    res = registry("uf", "tup-mid", env={**env, "LAUNCHER_OVERRIDE": str(fake)})
    assert res.returncode != 0
    assert "Missing input: " in res.stderr and hint in res.stderr
    assert (missing if missing.startswith("Skywork/") else str(tmp_path / "data" / missing)) in res.stderr
    assert not marker.exists()


@pytest.mark.parametrize("missing,hint", [
    ("shared/datasets-with-chosen-rewards/merged/magpieair-armorm/eval-checkpoint-selection-indices.npy",
     "bash scripts/download_data.sh"),
    ("shared/datasets/alpaca-ref-completions/mistral-sft/skywork_v2_llama", "bash scripts/score_reference_pools.sh"),
])
def test_registry_mistral_checks_its_own_split_and_reference_pools(tmp_path, missing, hint):
    env = make_inputs(tmp_path / "data", tmp_path / "hub", skip={missing}, inputs=MISTRAL_TUP_INPUTS)
    res = registry("mistral", "tup-mid", env=env)
    assert res.returncode != 0
    assert f"Missing input: {tmp_path / 'data' / missing}" in res.stderr and hint in res.stderr
    complete = make_inputs(tmp_path / "data2", tmp_path / "hub2", inputs=MISTRAL_TUP_INPUTS)
    res = registry("mistral", "tup-mid", env=complete)
    assert "Missing input" not in res.stderr and "TUP_DATA_DIR" in res.stderr


def test_registry_rejects_skywork_cache_at_another_revision(tmp_path):
    env = make_inputs(tmp_path / "data", tmp_path / "hub")
    ref = tmp_path / "hub/models--Skywork--Skywork-Reward-V2-Llama-3.1-8B/refs/main"
    ref.write_text("0" * 40)  # offline lookups would resolve a different snapshot than the pin
    res = registry("uf", "tup-mid", env=env)
    assert res.returncode != 0
    assert "Skywork/Skywork-Reward-V2-Llama-3.1-8B at revision cba2f842" in res.stderr
    assert "bash scripts/download_models.sh" in res.stderr


def test_registry_skip_flags_drop_the_matching_eval_input_checks(tmp_path):
    skipped = {"shared/datasets-with-chosen-rewards/merged/ultrafeedback-armorm/eval-reward-reporting-indices.npy",
               "shared/datasets/alpaca", "shared/datasets/alpaca-ref-completions/llama/armorm",
               *SKYWORK_PINS}
    env = make_inputs(tmp_path / "data", tmp_path / "hub", skip=skipped)
    fake = tmp_path / "fake_launcher.sh"
    fake.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake.chmod(0o755)
    res = registry("uf", "tup-mid", "--", "--skip-eval-split", "--skip-alpaca-eval", "--skip-rescore",
                   env={**env, "LAUNCHER_OVERRIDE": str(fake)})
    assert "Missing input" not in res.stderr
    assert "TUP_DATA_DIR" in res.stderr  # all inputs passed; only the data-dir location check remains


def test_registry_rejects_custom_data_dir_for_real_runs(tmp_path):
    # The eval stages of both launchers read $ROOT_DIR/data, so a custom TUP_DATA_DIR
    # may only be used by the download scripts and dry runs.
    env = make_inputs(tmp_path / "data", tmp_path / "hub")
    marker = tmp_path / "launcher_ran"
    fake = tmp_path / "fake_launcher.sh"
    fake.write_text(f'#!/usr/bin/env bash\ntouch {shlex.quote(str(marker))}\nexit 0\n')
    fake.chmod(0o755)
    res = registry("uf", "tup-mid", env={**env, "LAUNCHER_OVERRIDE": str(fake)})
    assert res.returncode != 0
    assert "TUP_DATA_DIR" in res.stderr
    assert not marker.exists()


def test_registry_data_dir_hint_links_an_absolute_path(tmp_path):
    # A relative TUP_DATA_DIR must not produce a relative (dangling) `ln -s` hint.
    env = make_inputs(tmp_path / "data", tmp_path / "hub")
    env["TUP_DATA_DIR"] = os.path.relpath(tmp_path / "data", ROOT)
    res = registry("uf", "tup-mid", env=env)
    assert res.returncode != 0
    real = os.path.realpath(tmp_path / "data")
    assert f'ln -s "{real}" "{ROOT}/data" && unset TUP_DATA_DIR' in res.stderr


def test_registry_accepts_tup_data_dir_linked_to_data(tmp_path):
    # After `ln -s "$TUP_DATA_DIR" data`, a real run must pass the data-dir check even while
    # TUP_DATA_DIR is still exported: both paths resolve to the same physical directory.
    real = tmp_path / "real"
    env = make_inputs(real, tmp_path / "hub")
    copy = tmp_path / "copy"  # the registry derives ROOT_DIR from its own location
    shutil.copytree(ROOT / "experiments", copy / "experiments")
    (copy / "scripts").mkdir()
    shutil.copy(ROOT / "scripts/pins.sh", copy / "scripts/pins.sh")
    os.symlink(real, copy / "data")
    marker = tmp_path / "launcher_ran"
    fake = tmp_path / "fake_launcher.sh"
    fake.write_text(f'#!/usr/bin/env bash\ntouch {shlex.quote(str(marker))}\nexit 0\n')
    fake.chmod(0o755)
    res = subprocess.run(["bash", str(copy / "experiments/llama_8b.sh"), "uf", "tup-mid"], cwd=copy,
                         capture_output=True, text=True,
                         env={**os.environ, **env, "LAUNCHER_OVERRIDE": str(fake)})
    assert res.returncode == 0, res.stderr
    assert marker.exists()


def test_registry_all_stops_on_first_failure(tmp_path):
    fake = tmp_path / "fake_launcher.sh"
    fake.write_text('#!/usr/bin/env bash\ncase " $* " in *" --loss-type qrpo "*) exit 7;; esac\nexit 0\n')
    fake.chmod(0o755)
    res = registry("uf", "all", "--dry-run", env={"LAUNCHER_OVERRIDE": str(fake)})
    assert res.returncode != 0
    assert "FAILED: uf/qrpo" in res.stderr
    assert "uf/qrpo-random" not in res.stdout  # nothing after the failure ran


def test_registry_exports_hf_hub_offline_to_launchers(tmp_path):
    # Offline mode makes the launchers' (and rescoring's) Skywork lookups resolve the pinned cached revision.
    out = tmp_path / "hf_hub_offline"
    fake = tmp_path / "fake_launcher.sh"
    fake.write_text(f'#!/usr/bin/env bash\nprintf %s "$HF_HUB_OFFLINE" > {shlex.quote(str(out))}\nexit 0\n')
    fake.chmod(0o755)
    res = registry("uf", "tup-mid", "--dry-run", env={"LAUNCHER_OVERRIDE": str(fake), "HF_HUB_OFFLINE": "0"})
    assert res.returncode == 0, res.stderr
    assert out.read_text() == "1"
