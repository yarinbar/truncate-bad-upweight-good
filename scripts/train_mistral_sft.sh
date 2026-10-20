#!/usr/bin/env bash
# Optional: retrain the Mistral-SFT start model exactly as for the paper (the hosted checkpoint is the default).
# Data: QRPO's mistral-nosft-magpieair offpolicy2best set, whose `chosen` column is the ArmoRM-best of the first
# 6 Mistral-7B-Instruct-v0.2 samples.
# Needs: bash scripts/download_models.sh --sft-base && bash scripts/download_data.sh --setting mistral-magpie --sft
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
DATA_DIR="${TUP_DATA_DIR:-$ROOT_DIR/data}"
DATASET_PATH="$DATA_DIR/shared/datasets-with-ref-rewards/merged/mistral-nosft-magpieair-armorm-temp1-ref50-offpolicy2best-armorm"
BASE_MODEL="$DATA_DIR/shared/models/mistral"
OUT_MODEL="$DATA_DIR/shared/models/mistral-sft"
DRY_RUN=0
for a in "$@"; do
  case "$a" in --dry-run) DRY_RUN=1 ;; *) echo "ERROR: unknown option $a (usage: bash scripts/train_mistral_sft.sh [--dry-run])" >&2; exit 1 ;; esac
done
CMD=(
  accelerate launch --config-file src/qrpo/configs/accelerate/ds-zero1.yaml
  --num_machines 1 --num_processes 4 --main_process_port "${MASTER_PORT:-29500}"
  -m qrpo.train_sft
  training_args.gradient_accumulation_steps=16 training_args.per_device_train_batch_size=2
  training_args.per_device_eval_batch_size=2 model=mistral dataset=magpieair-armorm
  "dataset_args.dataset_path=$DATASET_PATH" "model_args.model_name_or_path=$BASE_MODEL"
  training_args.num_train_epochs=1 training_args.max_grad_norm=10.0 training_args.learning_rate=5e-7
  job_subdir=sft-chosen/mistral-sft-magpieair-armorm outputs_subdir=shared resuming.resume=True
)
printf '+'; printf ' %q' "${CMD[@]}"; printf '\n'
[ "$DRY_RUN" = "1" ] && exit 0
[ -d "$DATASET_PATH" ] || { echo "ERROR: missing $DATASET_PATH; run: bash scripts/download_data.sh --setting mistral-magpie --sft" >&2; exit 1; }
[ -d "$BASE_MODEL" ] || { echo "ERROR: missing $BASE_MODEL; run: bash scripts/download_models.sh --sft-base" >&2; exit 1; }
[ ! -e "$OUT_MODEL" ] || { echo "ERROR: $OUT_MODEL exists; move it away to retrain" >&2; exit 1; }
"${CMD[@]}"
CKPT="$(find "$ROOT_DIR/outputs/shared/train_sft/sft-chosen/mistral-sft-magpieair-armorm/checkpoints" -maxdepth 2 -type d -name checkpoint-764 | head -1)"
[ -n "$CKPT" ] || { echo "ERROR: checkpoint-764 not found after training" >&2; exit 1; }
cp -r "$CKPT" "$OUT_MODEL"
echo "Wrote $OUT_MODEL"
