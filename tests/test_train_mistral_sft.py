import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDED = [
    "training_args.gradient_accumulation_steps=16", "training_args.per_device_train_batch_size=2",
    "training_args.per_device_eval_batch_size=2", "model=mistral", "dataset=magpieair-armorm",
    "training_args.num_train_epochs=1", "training_args.max_grad_norm=10.0", "training_args.learning_rate=5e-7",
    "job_subdir=sft-chosen/mistral-sft-magpieair-armorm", "outputs_subdir=shared", "resuming.resume=True",
]


def test_sft_dry_run_uses_recorded_overrides():
    res = subprocess.run(["bash", str(ROOT / "scripts/train_mistral_sft.sh"), "--dry-run"], cwd=ROOT,
                         capture_output=True, text=True, env={**os.environ, "TUP_DATA_DIR": "/d"})
    assert res.returncode == 0, res.stderr
    line = next(l for l in res.stdout.splitlines() if l.startswith("+ accelerate launch"))
    toks = shlex.split(line[2:])
    assert toks[toks.index("--config-file") + 1] == "src/qrpo/configs/accelerate/ds-zero1.yaml"
    assert toks[toks.index("--num_processes") + 1] == "4"
    args = toks[toks.index("qrpo.train_sft") + 1:]
    for override in RECORDED:
        assert override in args, override
    assert "dataset_args.dataset_path=/d/shared/datasets-with-ref-rewards/merged/mistral-nosft-magpieair-armorm-temp1-ref50-offpolicy2best-armorm" in args
    assert "model_args.model_name_or_path=/d/shared/models/mistral" in args
    assert not any(a.startswith(("training_args.seed=", "seed=")) for a in args)  # seed 42 comes from train-sft.yaml


def test_sft_rejects_unknown_options():
    # A typo such as --dryrun must not start a real 4-GPU training run.
    res = subprocess.run(["bash", str(ROOT / "scripts/train_mistral_sft.sh"), "--dryrun"], cwd=ROOT,
                         capture_output=True, text=True, env={**os.environ, "TUP_DATA_DIR": "/d"})
    assert res.returncode == 1
    assert "ERROR: unknown option --dryrun" in res.stderr
    assert "accelerate launch" not in res.stdout
