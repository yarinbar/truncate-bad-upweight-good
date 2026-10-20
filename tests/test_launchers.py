import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LLAMA = ["--dataset-id", "llama-nosft-ultrafeedback-armorm-temp1-ref50-offpolicy2random-armorm",
         "--loss-type", "ours", "--learning-rate", "1e-7", "--beta", "0.01", "--num-ref-rewards", "6",
         "--binomial-lambda", "0.5", "--qrpo-sample-selector", "random", "--use-discrete-partition", "False",
         "--model-path", "/x/models/llama", "--scheduler-max-steps", "476", "--max-steps", "400"]
MISTRAL = ["--dataset-id", "mistral-sft-magpieair-armorm-temp1-ref50-offpolicy2random-armorm",
           "--loss-type", "ours", "--learning-rate", "1e-6", "--beta", "0.01", "--num-ref-rewards", "6",
           "--binomial-lambda", "0.5", "--qrpo-sample-selector", "random", "--use-discrete-partition", "False",
           "--model-path", "/x/models/mistral-sft", "--ref-model-key", "mistral-sft",
           "--scheduler-max-steps", "764", "--max-steps", "764"]


def run(script, args, **env):
    full_env = {**os.environ, "DRY_RUN": "1", **env}
    return subprocess.run(["bash", str(ROOT / script), *args], cwd=ROOT, env=full_env,
                          capture_output=True, text=True)


def train_overrides(stdout):
    line = next(l for l in stdout.splitlines() if l.startswith("+ accelerate launch"))
    toks = shlex.split(line[2:])
    return toks, dict(t.split("=", 1) for t in toks[toks.index("qrpo.train_qrpo") + 1:] if "=" in t)


@pytest.mark.parametrize("script,args,eval_bs,ref_key", [
    ("run_llama_experiment.sh", LLAMA, "4", "llama"),
    ("run_mistral_experiment.sh", MISTRAL, "8", "mistral-sft"),
])
def test_dry_run_prints_full_train_command_without_gpus(script, args, eval_bs, ref_key):
    res = run(script, args)
    assert res.returncode == 0, res.stderr
    toks, ov = train_overrides(res.stdout)
    assert toks[toks.index("--num_processes") + 1] == "4"
    assert ov["training_args.max_steps"] in {"476", "764"}
    assert ov["stop_after_steps"] == args[args.index("--max-steps") + 1]
    assert ov["training_args.per_device_eval_batch_size"] == eval_bs
    assert ov["++ref_model_key"] == ref_key
    assert ov["model_args.model_name_or_path"] == args[args.index("--model-path") + 1]
    assert "RUN_ROOT=" in res.stdout


@pytest.mark.parametrize("script,args", [("run_llama_experiment.sh", LLAMA), ("run_mistral_experiment.sh", MISTRAL)])
def test_scheduler_max_steps_is_required(script, args):
    i = args.index("--scheduler-max-steps")
    res = run(script, args[:i] + args[i + 2:])
    assert res.returncode != 0
    assert "--scheduler-max-steps is required" in res.stderr
    usage = run(script, ["--help"]).stdout
    assert "    --scheduler-max-steps <scheduler_max_steps> \\" in usage.splitlines()  # required, so no brackets


def test_llama_magpie_never_auto_picks_sft_chosen_checkpoint():
    text = (ROOT / "run_llama_experiment.sh").read_text()
    assert "sft-chosen" not in text


@pytest.mark.parametrize("script,args", [("run_llama_experiment.sh", LLAMA), ("run_mistral_experiment.sh", MISTRAL)])
def test_eval_base_only_dry_run(script, args):
    res = run(script, [*args, "--eval-base-only"])
    assert res.returncode == 0, res.stderr
    assert "+ accelerate launch" not in res.stdout
    line = next(l for l in res.stdout.splitlines() if l.startswith("+ eval-base-only "))
    assert line.split()[-1].endswith("/chat-baseline/initial-" + args[args.index("--dataset-id") + 1])


# --- non-dry --eval-base-only runs in a sandbox copy of the launchers (fake python/accelerate) ---

FAKE_PYTHON = r"""#!/usr/bin/env bash
# Fake interpreter for launcher preflight and for qrpo.evals.run_online_eval.
if [ "$1" = "-m" ] && [ "$2" = "qrpo.evals.run_online_eval" ]; then
  shift 2
  echo "$*" >> "$FAKE_LOG.argv"
  steps=""
  for a in "$@"; do
    case "$a" in
      training_dir=*) td="${a#training_dir=}" ;;
      eval_type=*) et="${a#eval_type=}" ;;
      model_args.model_name_or_path=*) mp="${a#model_args.model_name_or_path=}" ;;
      checkpoint_steps=*) steps="${a#checkpoint_steps=}" ;;
    esac
  done
  hash_dir="$(ls -d "$td"/checkpoints/*/ | head -1)"
  hash_dir="${hash_dir%/}"
  # Mirror run_online_eval.py: a missing checkpoint-0 becomes a symlink to the base model dir.
  if [ ! -e "$hash_dir/checkpoint-0" ]; then
    ln -s "$mp" "$hash_dir/checkpoint-0"
  fi
  # Mirror its checkpoint selection: checkpoint_steps=[N] evaluates only checkpoint-N, no filter evaluates
  # every checkpoint (FAKE_EVAL_STEP simulates an eval that wrote its outputs to another checkpoint).
  step="${steps#[}"
  step="${step%]}"
  step="${FAKE_EVAL_STEP:-$step}"
  if [ -n "$step" ]; then
    targets=""
    for n in ${step//,/ }; do
      targets="$targets $hash_dir/checkpoint-$n"
    done
  else
    targets="$(ls -d "$hash_dir"/checkpoint-*)"
  fi
  for ck in $targets; do
    if [ "$(basename "$ck")" = "checkpoint-0" ]; then
      echo "PRE $et $(ls -A "$hash_dir/checkpoint-0" | sort | tr '\n' ' ')" >> "$FAKE_LOG"
      echo "CK0 $td $et $(cd -P "$hash_dir/checkpoint-0" && pwd)" >> "$FAKE_LOG.ck0"
    fi
    mkdir -p "$ck/online_evals_0/$et/kl-dataset"
  done
  exit 0
fi
src="$(cat)"
if [[ "$src" == *initialize_config_dir* ]]; then
  # The launcher's config-composing snippet runs for real (Hydra, CPU only).
  shift  # drop the leading "-" (script on stdin)
  exec "$FAKE_REAL_PYTHON" - "$@" <<< "$src"
fi
echo 4
exit 0
"""

BASE_MODEL_FILES = ["config.json", "model.safetensors", "tokenizer.json"]


def make_sandbox(tmp_path, script, args):
    sandbox = tmp_path / "repo"
    sandbox.mkdir()
    (sandbox / script).write_bytes((ROOT / script).read_bytes())
    dataset_id = args[args.index("--dataset-id") + 1]
    (sandbox / "data/shared/datasets-with-ref-rewards/merged" / dataset_id).mkdir(parents=True)
    (sandbox / "data/shared/datasets/alpaca").mkdir(parents=True)
    (sandbox / "src").symlink_to(ROOT / "src", target_is_directory=True)

    model = tmp_path / "base_model"
    model.mkdir()
    for name in BASE_MODEL_FILES:
        (model / name).write_text(name)
    (model / "online_evals_0").mkdir()
    (model / "online_evals_0" / "stale.txt").write_text("stale")
    (model / "online_evals_logs_eval_split.json").write_text("{}")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("python", FAKE_PYTHON), ("accelerate", "#!/usr/bin/env bash\nexit 0\n"),
                       ("nvidia-smi", "#!/usr/bin/env bash\nexit 0\n")):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)

    env = {k: v for k, v in os.environ.items() if k not in {"DRY_RUN", "CHECK_ONLY"}}
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["FAKE_LOG"] = str(tmp_path / "fake.log")
    env["FAKE_REAL_PYTHON"] = sys.executable
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    run_root = sandbox / "outputs/shared/train_qrpo/chat-baseline" / f"initial-{dataset_id}"
    return sandbox, model, env, run_root


def run_base_only(sandbox, script, args, model, env):
    full = [*args[:args.index("--model-path")], *args[args.index("--model-path") + 2:],
            "--model-path", str(model), "--eval-base-only", "--skip-rescore"]
    return subprocess.run(["bash", str(sandbox / script), *full], cwd=sandbox, env=env,
                          capture_output=True, text=True)


@pytest.mark.parametrize("script,args", [("run_llama_experiment.sh", LLAMA), ("run_mistral_experiment.sh", MISTRAL)])
def test_eval_base_only_gives_each_run_its_own_checkpoint_0(tmp_path, script, args):
    sandbox, model, env, run_root = make_sandbox(tmp_path, script, args)
    res = run_base_only(sandbox, script, args, model, env)
    assert res.returncode == 0, res.stdout + res.stderr

    checkpoint_0 = run_root / "checkpoints/base/checkpoint-0"
    assert checkpoint_0.is_dir() and not checkpoint_0.is_symlink()
    for name in BASE_MODEL_FILES:
        link = checkpoint_0 / name
        assert link.is_symlink()
        assert os.readlink(link) == str(model / name)
    # Eval outputs live in the per-run dir only; nothing leaked from or into the shared model dir.
    assert not (checkpoint_0 / "online_evals_logs_eval_split.json").exists()
    assert not (checkpoint_0 / "online_evals_0" / "stale.txt").exists()
    assert (checkpoint_0 / "online_evals_0" / "eval_split" / "kl-dataset").is_dir()
    assert sorted(p.name for p in model.iterdir()) == sorted(
        [*BASE_MODEL_FILES, "online_evals_0", "online_evals_logs_eval_split.json"])
    assert [p.name for p in (model / "online_evals_0").iterdir()] == ["stale.txt"]
    # Before the first eval ran, checkpoint-0 held exactly the model files (no stale outputs to reuse).
    pre = [l.split(maxsplit=2) for l in (tmp_path / "fake.log").read_text().splitlines()]
    assert [p[1] for p in pre] == ["eval_split", "alpaca_eval"]
    assert pre[0][2].split() == BASE_MODEL_FILES

    # A rerun resumes against the same real directory.
    res = run_base_only(sandbox, script, args, model, env)
    assert res.returncode == 0, res.stdout + res.stderr


@pytest.mark.parametrize("script,args", [("run_llama_experiment.sh", LLAMA), ("run_mistral_experiment.sh", MISTRAL)])
def test_eval_base_only_refuses_symlinked_checkpoint_0(tmp_path, script, args):
    sandbox, model, env, run_root = make_sandbox(tmp_path, script, args)
    base = run_root / "checkpoints/base"
    base.mkdir(parents=True)
    (base / "checkpoint-0").symlink_to(model, target_is_directory=True)
    res = run_base_only(sandbox, script, args, model, env)
    assert res.returncode != 0
    assert "symlinked base checkpoint" in res.stderr
    assert (base / "checkpoint-0").is_symlink()
    assert [p.name for p in (model / "online_evals_0").iterdir()] == ["stale.txt"]


@pytest.mark.parametrize("script,args,model_key,dataset_name", [
    ("run_llama_experiment.sh", LLAMA, "llama", "ultrafeedback-armorm"),
    ("run_mistral_experiment.sh", MISTRAL, "mistral", "magpieair-armorm"),
])
def test_eval_base_only_writes_config_that_rescoring_can_discover(tmp_path, script, args, model_key, dataset_name):
    from rescore_completions import build_eval_cmd, discover_training_runs

    sandbox, model, env, run_root = make_sandbox(tmp_path, script, args)

    # DRY_RUN must create nothing at all (no outputs/, no config).
    dry = subprocess.run(["bash", str(sandbox / script), *args, "--eval-base-only"], cwd=sandbox,
                         env={**env, "DRY_RUN": "1"}, capture_output=True, text=True)
    assert dry.returncode == 0, dry.stderr
    assert not (sandbox / "outputs").exists()

    res = run_base_only(sandbox, script, args, model, env)
    assert res.returncode == 0, res.stdout + res.stderr

    config_path = run_root / "checkpoints/base/config/process-0/config-postprocessed.yaml"
    assert config_path.is_file()
    assert "${" not in config_path.read_text()
    assert not config_path.with_name(config_path.name + ".tmp").exists()

    runs = discover_training_runs(run_root)
    assert len(runs) == 1
    training_dir, saved_config, checkpoints = runs[0]
    assert Path(training_dir) == run_root
    assert [c.name for c in checkpoints] == ["checkpoint-0"]
    assert "hydra" not in saved_config

    dataset_id = args[args.index("--dataset-id") + 1]
    dataset_path = str(sandbox / "data/shared/datasets-with-ref-rewards/merged" / dataset_id)
    cmd = build_eval_cmd(saved_config, training_dir, "skywork_v2_llama_", "eval_split",
                         model_name_override=model_key, dataset_name_override=dataset_name,
                         ref_model_key_override=args[args.index("--ref-model-key") + 1]
                         if "--ref-model-key" in args else model_key)
    assert f"dataset_args.dataset_path={dataset_path}" in cmd
    assert f"model_args.model_name_or_path={model}" in cmd
    assert f"data_dir={sandbox / 'data'}" in cmd
    assert f"job_subdir=chat-baseline/initial-{dataset_id}" in cmd
    assert "training_args.loss_type=ours" in cmd
    assert "training_args.num_ref_rewards=6" in cmd
    assert "training_args.beta=0.01" in cmd
    assert "training_args.qrpo_sample_selector=random" in cmd
    assert saved_config["training_args"]["binomial_lambda"] == 0.5
    assert saved_config["training_args"]["use_discrete_partition"] is False

    # A rerun keeps the existing config untouched.
    before = config_path.read_bytes()
    assert run_base_only(sandbox, script, args, model, env).returncode == 0
    assert config_path.read_bytes() == before


# --- trained runs: only the stop checkpoint is evaluated and rescored ---

FAKE_ACCELERATE = r"""#!/usr/bin/env bash
# Fake training: create the final checkpoint the launcher expects (cwd is the launcher's ROOT_DIR).
for a in "$@"; do
  case "$a" in
    job_subdir=*) js="${a#job_subdir=}" ;;
    stop_after_steps=*) stop="${a#stop_after_steps=}" ;;
  esac
done
mkdir -p "outputs/shared/train_qrpo/$js/checkpoints/h0/checkpoint-$stop"
touch "outputs/shared/train_qrpo/$js/checkpoints/h0/checkpoint-$stop/model.safetensors"
"""

FAKE_RESCORE = r"""#!/usr/bin/env bash
# Fake run_rescore_multi_rm.sh: record the invocation, then create the outputs the launcher checks
# (FAKE_RESCORE_STEP simulates a rescoring that wrote its outputs to another checkpoint).
run_root="$(head -1 "$DIRS_FILE")"
hash_dir="$(ls -d "$run_root"/checkpoints/*/ | head -1)"
hash_dir="${hash_dir%/}"
echo "CHECKPOINT_STEPS=$CHECKPOINT_STEPS|EVAL_TYPES=$EVAL_TYPES|RMS=$*|DIRS=$(tr '\n' ' ' < "$DIRS_FILE" | sed 's/ $//')" >> "$FAKE_LOG.rescore"
steps="${FAKE_RESCORE_STEP:-$CHECKPOINT_STEPS}"
if [ -z "$steps" ]; then
  steps="$(cd "$hash_dir" && ls -d checkpoint-* | sed 's/^checkpoint-//' | tr '\n' ' ')"
fi
for rm in "$@"; do
  prefix="${rm}_"
  for s in $steps; do
    mkdir -p "$hash_dir/checkpoint-$s/${prefix}online_evals_0"
    for t in $EVAL_TYPES; do
      echo '{}' > "$hash_dir/checkpoint-$s/${prefix}online_evals_logs_$t.json"
    done
  done
done
"""


def with_arg(args, flag, value):
    out = list(args)
    out[out.index(flag) + 1] = value
    return out


def make_trained_sandbox(tmp_path, script, args):
    sandbox, model, env, _ = make_sandbox(tmp_path, script, args)
    (tmp_path / "bin" / "accelerate").write_text(FAKE_ACCELERATE)
    (sandbox / "run_rescore_multi_rm.sh").write_text(FAKE_RESCORE)
    dataset_id = args[args.index("--dataset-id") + 1]
    for rm_name in ("skywork_v2_llama", "skywork_v2_qwen"):
        (sandbox / "data/shared/datasets-with-ref-rewards/merged" / dataset_id
         / "eval-split-rewards" / rm_name).mkdir(parents=True)
    return sandbox, model, env


def run_trained(sandbox, script, args, model, env, *extra, dry=False):
    full = [*with_arg(args, "--model-path", str(model)), *extra]
    return subprocess.run(["bash", str(sandbox / script), *full], cwd=sandbox,
                          env={**env, "DRY_RUN": "1"} if dry else env, capture_output=True, text=True)


def run_root_of(sandbox, script, args, model, env, *extra):
    dry = run_trained(sandbox, script, args, model, env, *extra, dry=True)
    assert dry.returncode == 0, dry.stderr
    return Path(next(l.split("=", 1)[1] for l in dry.stdout.splitlines() if l.startswith("RUN_ROOT=")))


def eval_calls(tmp_path):
    # One (eval_type, [checkpoint_steps override, ...]) per run_online_eval invocation, in order.
    calls = []
    for line in (tmp_path / "fake.log.argv").read_text().splitlines():
        toks = line.split()
        eval_type = next(t.split("=", 1)[1] for t in toks if t.startswith("eval_type="))
        calls.append((eval_type, [t for t in toks if t.startswith("checkpoint_steps=")]))
    return calls


def rescore_calls(tmp_path):
    path = tmp_path / "fake.log.rescore"
    return [dict(f.split("=", 1) for f in line.split("|")) for line in path.read_text().splitlines()]


def assert_only_steps_evaluated_and_rescored(tmp_path, sandbox, model, run_root, steps, runs=1):
    steps = [str(step) for step in (steps if isinstance(steps, (list, tuple)) else [steps])]
    hash_dir = run_root / "checkpoints/h0"
    # Both eval types were asked for exactly these checkpoints (never checkpoint-0), nothing else.
    override = [f"checkpoint_steps=[{','.join(steps)}]"]
    assert eval_calls(tmp_path) == [("eval_split", override), ("alpaca_eval", override)] * runs
    # One rescoring call per run, restricted to the same checkpoints.
    rescoring = rescore_calls(tmp_path)
    assert len(rescoring) == runs
    for call in rescoring:
        assert call == {"CHECKPOINT_STEPS": " ".join(steps), "EVAL_TYPES": "eval_split alpaca_eval",
                        "RMS": "skywork_v2_llama skywork_v2_qwen", "DIRS": str(run_root)}
    for step in steps:
        for eval_type in ("eval_split", "alpaca_eval"):
            assert (hash_dir / f"checkpoint-{step}/online_evals_0/{eval_type}/kl-dataset").is_dir()
            assert (hash_dir / f"checkpoint-{step}/skywork_v2_llama_online_evals_logs_{eval_type}.json").is_file()
        assert (hash_dir / f"checkpoint-{step}/skywork_v2_qwen_online_evals_0").is_dir()
    # The base model (checkpoint-0) was not evaluated: no per-dataset base eval dir, nothing written to the
    # shared model dir that run_online_eval.py links as checkpoint-0.
    assert not (sandbox / "outputs/shared/train_qrpo/base-model-evals").exists()
    assert not (tmp_path / "fake.log.ck0").exists()
    assert sorted(p.name for p in model.iterdir()) == sorted(
        [*BASE_MODEL_FILES, "online_evals_0", "online_evals_logs_eval_split.json"])
    assert [p.name for p in (model / "online_evals_0").iterdir()] == ["stale.txt"]


@pytest.mark.parametrize("script,args", [("run_llama_experiment.sh", LLAMA), ("run_mistral_experiment.sh", MISTRAL)])
def test_trained_run_evaluates_and_rescores_only_the_stop_checkpoint(tmp_path, script, args):
    sandbox, model, env = make_trained_sandbox(tmp_path, script, args)
    max_steps = args[args.index("--max-steps") + 1]

    assert run_trained(sandbox, script, args, model, env, dry=True).returncode == 0
    assert not (sandbox / "outputs").exists()  # DRY_RUN creates nothing

    run_root = run_root_of(sandbox, script, args, model, env)
    res = run_trained(sandbox, script, args, model, env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert_only_steps_evaluated_and_rescored(tmp_path, sandbox, model, run_root, max_steps)

    # A rerun (training done, checkpoints pruned again) evaluates and rescores the same checkpoint again.
    res = run_trained(sandbox, script, args, model, env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert_only_steps_evaluated_and_rescored(tmp_path, sandbox, model, run_root, max_steps, runs=2)


@pytest.mark.parametrize("script,args", [("run_llama_experiment.sh", LLAMA), ("run_mistral_experiment.sh", MISTRAL)])
@pytest.mark.parametrize("fake_var,missing", [
    ("FAKE_EVAL_STEP", "Missing eval output: {hash}/checkpoint-{step}/online_evals_0/eval_split/kl-dataset"),
    ("FAKE_RESCORE_STEP", "Missing rescored online eval output: {hash}/checkpoint-{step}/skywork_v2_llama_online_evals_0"),
], ids=["eval", "rescore"])
def test_trained_run_checks_the_outputs_of_its_stop_checkpoint(tmp_path, script, args, fake_var, missing):
    sandbox, model, env = make_trained_sandbox(tmp_path, script, args)
    max_steps = args[args.index("--max-steps") + 1]
    run_root = run_root_of(sandbox, script, args, model, env)
    # Outputs that landed in some other checkpoint do not satisfy the check.
    res = run_trained(sandbox, script, args, model, {**env, fake_var: "1"})
    assert res.returncode != 0
    assert missing.format(hash=run_root / "checkpoints/h0", step=max_steps) in res.stderr


def test_llama_explicit_checkpoint_step_wins_over_max_steps(tmp_path):
    script, args = "run_llama_experiment.sh", LLAMA
    sandbox, model, env = make_trained_sandbox(tmp_path, script, args)
    extra = ("--keep-all-checkpoints", "--checkpoint-step", "160")
    run_root = run_root_of(sandbox, script, args, model, env, *extra)
    hash_dir = run_root / "checkpoints/h0"
    for step in ("160", "400"):
        (hash_dir / f"checkpoint-{step}").mkdir(parents=True)  # a kept run: no training happens

    res = run_trained(sandbox, script, args, model, env, *extra)
    assert res.returncode == 0, res.stdout + res.stderr
    assert_only_steps_evaluated_and_rescored(tmp_path, sandbox, model, run_root, 160)
    assert not (hash_dir / "checkpoint-400/online_evals_0").exists()


@pytest.mark.parametrize("zero_kind", ["symlink", "dir"])
@pytest.mark.parametrize("script,args", [("run_llama_experiment.sh", LLAMA), ("run_mistral_experiment.sh", MISTRAL)])
def test_keep_all_checkpoints_without_a_step_evaluates_every_kept_training_step(tmp_path, script, args, zero_kind):
    sandbox, model, env = make_trained_sandbox(tmp_path, script, args)
    max_steps = args[args.index("--max-steps") + 1]
    extra = ("--keep-all-checkpoints",)
    run_root = run_root_of(sandbox, script, args, model, env, *extra)
    hash_dir = run_root / "checkpoints/h0"
    for step in ("160", "80"):  # intermediate checkpoints kept from a resumed run
        (hash_dir / f"checkpoint-{step}").mkdir(parents=True, exist_ok=True)
    if zero_kind == "symlink":  # what run_online_eval.py leaves behind for the base model
        (hash_dir / "checkpoint-0").symlink_to(model, target_is_directory=True)
    else:
        (hash_dir / "checkpoint-0").mkdir()
    (hash_dir / "checkpoint-9").symlink_to(hash_dir / "checkpoint-80", target_is_directory=True)  # symlinks are skipped

    res = run_trained(sandbox, script, args, model, env, *extra)
    assert res.returncode == 0, res.stdout + res.stderr
    # Every kept training step, in numeric order (80 before 160), never checkpoint-0 or a symlink; the rescoring
    # call gets the same steps as a space separated list.
    assert_only_steps_evaluated_and_rescored(tmp_path, sandbox, model, run_root, ["80", "160", max_steps])
    if zero_kind == "dir":  # (a symlinked checkpoint-0 is the shared model dir, checked by the helper)
        assert not (hash_dir / "checkpoint-0/online_evals_0").exists()
    assert f"Evaluating kept checkpoints (steps 80 160 {max_steps})" in res.stdout


@pytest.mark.parametrize("script,args", [("run_llama_experiment.sh", LLAMA), ("run_mistral_experiment.sh", MISTRAL)])
def test_keep_all_checkpoints_requires_the_stop_checkpoint(tmp_path, script, args):
    sandbox, model, env = make_trained_sandbox(tmp_path, script, args)
    (tmp_path / "bin" / "accelerate").write_text("#!/usr/bin/env bash\nexit 0\n")  # training makes no checkpoint
    extra = ("--keep-all-checkpoints",)
    run_root = run_root_of(sandbox, script, args, model, env, *extra)
    (run_root / "checkpoints/h0/checkpoint-160").mkdir(parents=True)

    res = run_trained(sandbox, script, args, model, env, *extra)
    assert res.returncode != 0
    assert "final checkpoint is missing" in res.stderr
    assert not (tmp_path / "fake.log.argv").exists()  # no eval ran


@pytest.mark.parametrize("script,args", [("run_llama_experiment.sh", LLAMA), ("run_mistral_experiment.sh", MISTRAL)])
def test_eval_base_only_still_evaluates_its_own_checkpoint_0_without_a_step_filter(tmp_path, script, args):
    sandbox, model, env = make_trained_sandbox(tmp_path, script, args)
    dataset_id = args[args.index("--dataset-id") + 1]
    run_root = sandbox / "outputs/shared/train_qrpo/chat-baseline" / f"initial-{dataset_id}"
    res = run_trained(sandbox, script, args, model, env, "--eval-base-only")
    assert res.returncode == 0, res.stdout + res.stderr

    assert eval_calls(tmp_path) == [("eval_split", []), ("alpaca_eval", [])]
    assert rescore_calls(tmp_path) == [{"CHECKPOINT_STEPS": "", "EVAL_TYPES": "eval_split alpaca_eval",
                                        "RMS": "skywork_v2_llama skywork_v2_qwen", "DIRS": str(run_root)}]
    checkpoint_0 = run_root / "checkpoints/base/checkpoint-0"
    assert checkpoint_0.is_dir() and not checkpoint_0.is_symlink()
    assert (checkpoint_0 / "online_evals_0/eval_split/kl-dataset").is_dir()
    assert (checkpoint_0 / "skywork_v2_llama_online_evals_0").is_dir()
    assert not (sandbox / "outputs/shared/train_qrpo/base-model-evals").exists()
