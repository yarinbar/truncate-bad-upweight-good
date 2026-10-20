import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def judge(*args, env=None):
    return subprocess.run(["bash", str(ROOT / "scripts/judge_alpaca_gpt4o.sh"), *args], cwd=ROOT,
                          capture_output=True, text=True, env={**os.environ, **(env or {})})


def test_judge_requires_generations_first(tmp_path):
    ckpt = tmp_path / "checkpoint-400"
    ckpt.mkdir()
    res = judge(str(ckpt), env={"OPENAI_API_KEY": ""})
    assert res.returncode != 0
    assert "online_evals_0/alpaca_eval/model_outputs.json" in res.stderr
    assert "OPENAI_API_KEY" not in res.stderr  # generation check comes before the key check


def test_judge_dry_run_uses_gpt4o_config(tmp_path):
    ckpt = tmp_path / "checkpoint-400"
    (ckpt / "online_evals_0/alpaca_eval").mkdir(parents=True)
    (ckpt / "online_evals_0/alpaca_eval/model_outputs.json").write_text("[]")
    res = judge(str(ckpt), "--dry-run")
    assert res.returncode == 0, res.stderr
    assert "alpaca_eval_gpt4o_fn" in res.stdout and "WANDB_MODE=disabled" in res.stdout
    assert "--search-root" in res.stdout


def test_judge_command_is_accepted_by_upstream_dry_run(tmp_path):
    # Checkpoint lives outside outputs/, so upstream's default search root does not exist.
    ckpt = tmp_path / "ckpt-outside/checkpoint-400"
    (ckpt / "online_evals_0/alpaca_eval").mkdir(parents=True)
    (ckpt / "online_evals_0/alpaca_eval/model_outputs.json").write_text("[]")
    res = judge(str(ckpt), "--dry-run")
    assert res.returncode == 0, res.stderr
    line = next(l for l in res.stdout.splitlines() if l.startswith("+ "))
    args = shlex.split(line[2:])
    # Replace the wrapper's temp list file (already deleted) with one we recreate.
    list_file = tmp_path / "list.txt"
    list_file.write_text(f"{ckpt.resolve()}\n")
    args[args.index("--from-file") + 1] = str(list_file)
    assert "--dry-run" not in args
    args.append("--dry-run")  # never run upstream for real in tests
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    up = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, env=env)
    assert up.returncode == 0, up.stdout + up.stderr
    assert "Checkpoints to score: 1" in up.stdout
    assert "alpaca_eval_gpt4o_fn -> gpt-4o" in up.stdout
