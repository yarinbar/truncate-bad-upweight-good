#!/usr/bin/env bash
set -euo pipefail

# Re-score completions with multiple reward models.
#
# This script also ensures dataset-local eval-split reference rewards exist at:
#   <dataset_path>/eval-split-rewards/<rm_name>/
#
# Usage:
#   bash run_rescore_multi_rm.sh                         # default: both Skywork V2 RMs
#   bash run_rescore_multi_rm.sh skywork_v2_llama
#   bash run_rescore_multi_rm.sh armorm                 # optional, usually unnecessary
#   OVERWRITE_REF_REWARDS=1 bash run_rescore_multi_rm.sh skywork_v2_qwen
#   CHECKPOINT_STEPS="480" bash run_rescore_multi_rm.sh skywork_v2_llama
#   CHECKPOINT_STEPS="160 480" bash run_rescore_multi_rm.sh skywork_v2_llama

DIRS_FILE="${DIRS_FILE:-rescore_dirs.txt}"
EVAL_TYPES="${EVAL_TYPES:-alpaca_eval eval_split}"
NUM_SEEDS="${NUM_SEEDS:-3}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
OVERWRITE_REF_REWARDS="${OVERWRITE_REF_REWARDS:-0}"
RESCORE_MODEL="${RESCORE_MODEL:-}"
RESCORE_DATASET="${RESCORE_DATASET:-}"
RESCORE_REF_MODEL_KEY="${RESCORE_REF_MODEL_KEY:-}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-}"
declare -a CHECKPOINT_STEP_VALUES=()
read -r -a CHECKPOINT_STEP_VALUES <<< "$CHECKPOINT_STEPS"
CHECKPOINT_ARGS=()
if [ "${#CHECKPOINT_STEP_VALUES[@]}" -gt 0 ]; then
  CHECKPOINT_ARGS=(--checkpoint_steps "${CHECKPOINT_STEP_VALUES[@]}")
fi

if [ -z "$RESCORE_MODEL" ] || [ -z "$RESCORE_DATASET" ] || [ -z "$RESCORE_REF_MODEL_KEY" ]; then
  echo "ERROR: RESCORE_MODEL, RESCORE_DATASET, and RESCORE_REF_MODEL_KEY are required." >&2
  exit 1
fi

if [ ! -f "$DIRS_FILE" ]; then
  echo "ERROR: $DIRS_FILE not found."
  exit 1
fi

mapfile -t TRAINING_DIRS < <(sed '/^$/d' "$DIRS_FILE")

RESCORE_EVAL_OVERRIDES=()
if [ -n "$RESCORE_MODEL" ]; then
  RESCORE_EVAL_OVERRIDES+=(--model "$RESCORE_MODEL")
fi
if [ -n "$RESCORE_DATASET" ]; then
  RESCORE_EVAL_OVERRIDES+=(--dataset "$RESCORE_DATASET")
fi
if [ -n "$RESCORE_REF_MODEL_KEY" ]; then
  RESCORE_EVAL_OVERRIDES+=(--ref_model_key "$RESCORE_REF_MODEL_KEY")
fi

declare -A MODELS
MODELS[armorm]="data/shared/models/armorm"
MODELS[skywork_v2_llama]="Skywork/Skywork-Reward-V2-Llama-3.1-8B"
MODELS[skywork_v2_qwen]="Skywork/Skywork-Reward-V2-Qwen3-8B"

declare -A PREFIXES
PREFIXES[armorm]="armorm_"
PREFIXES[skywork_v2_llama]="skywork_v2_llama_"
PREFIXES[skywork_v2_qwen]="skywork_v2_qwen_"

if [ $# -gt 0 ]; then
  RUN_KEYS=("$@")
else
  RUN_KEYS=(skywork_v2_llama skywork_v2_qwen)
fi

for key in "${RUN_KEYS[@]}"; do
  if [ -z "${MODELS[$key]+x}" ]; then
    echo "ERROR: Unknown model key '$key'. Available: ${!MODELS[*]}"
    exit 1
  fi
done

check_phase1_complete() {
  local training_dir="$1"
  local prefix="$2"

  local checkpoints_parent="$training_dir/checkpoints"
  local hash_dir
  hash_dir=$(ls "$checkpoints_parent" 2>/dev/null | head -1)
  [ -z "$hash_dir" ] && return 1

  local -a checkpoint_dirs=()
  if [ "${#CHECKPOINT_STEP_VALUES[@]}" -gt 0 ]; then
    local checkpoint_dir
    local step
    for step in "${CHECKPOINT_STEP_VALUES[@]}"; do
      checkpoint_dir="$checkpoints_parent/$hash_dir/checkpoint-$step"
      [ -d "$checkpoint_dir" ] || return 1
      checkpoint_dirs+=("$checkpoint_dir")
    done
  else
    checkpoint_dirs=("$checkpoints_parent/$hash_dir"/checkpoint-*)
  fi

  local ckpt_dir
  for ckpt_dir in "${checkpoint_dirs[@]}"; do
    [ ! -d "$ckpt_dir" ] && continue
    for seed in $(seq 0 $((NUM_SEEDS - 1))); do
      for eval_type in $EVAL_TYPES; do
        local kl_path="$ckpt_dir/${prefix}online_evals_${seed}/${eval_type}/kl-dataset"
        if [ ! -d "$kl_path" ]; then
          return 1
        fi
      done
    done
  done
  return 0
}

discover_eval_split_datasets() {
  python - "$DIRS_FILE" <<'PY'
import sys
from pathlib import Path

import yaml

dirs_file = Path(sys.argv[1])
seen = set()

for raw_line in dirs_file.read_text().splitlines():
    line = raw_line.strip()
    if not line:
        continue
    root = Path(line)
    config_paths = sorted(
        root.rglob("checkpoints/*/config/process-0/config-postprocessed.yaml")
    )
    if not config_paths:
        continue
    for config_path in config_paths:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        dataset_path = config["dataset_args"]["dataset_path"]
        split_name = config["dataset_args"]["eval_split"]["name"]
        key = (dataset_path, split_name)
        if key in seen:
            continue
        seen.add(key)
        print(f"{dataset_path}\t{split_name}")
PY
}

ensure_eval_split_ref_rewards() {
  local rm_name="$1"
  local rm_path="$2"

  if [[ " $EVAL_TYPES " != *" eval_split "* ]]; then
    return
  fi

  mapfile -t dataset_entries < <(discover_eval_split_datasets)
  if [ "${#dataset_entries[@]}" -eq 0 ]; then
    echo "No eval-split datasets discovered from $DIRS_FILE"
    return
  fi

  echo ""
  echo "---- Ensuring dataset-local eval-split ref rewards for $rm_name ----"
  for entry in "${dataset_entries[@]}"; do
    dataset_path="${entry%%$'\t'*}"
    split_name="${entry#*$'\t'}"
    output_path="$dataset_path/eval-split-rewards/$rm_name"

    if [ -d "$output_path" ] && [ "$OVERWRITE_REF_REWARDS" != "1" ]; then
      echo "Skipping existing $output_path"
      continue
    fi

    echo "Scoring $dataset_path [$split_name] -> $output_path"
    cmd=(
      python score_eval_split_ref_completions.py
      --dataset_path "$dataset_path"
      --split "$split_name"
      --rm_name "$rm_name"
      --rm_path "$rm_path"
      --num_gpus "$NUM_GPUS"
      --batch_size "$BATCH_SIZE"
      --max_seq_length "$MAX_SEQ_LENGTH"
    )
    if [ "$OVERWRITE_REF_REWARDS" = "1" ]; then
      cmd+=(--overwrite)
    fi
    "${cmd[@]}"
  done
}

SELECTED_MODE_FAILED=0
for key in "${RUN_KEYS[@]}"; do
  REWARD_MODEL="${MODELS[$key]}"
  PREFIX="${PREFIXES[$key]}"
  PHASE1_FAILED=()
  PHASE2_FAILED=()
  PHASE2_SKIPPED=()

  echo ""
  echo "============================================================"
  echo "  MODEL: $REWARD_MODEL"
  echo "  PREFIX: $PREFIX"
  echo "============================================================"

  ensure_eval_split_ref_rewards "$key" "$REWARD_MODEL"

  echo ""
  echo "---- Phase 1: Scoring ${#TRAINING_DIRS[@]} experiments ----"
  for i in "${!TRAINING_DIRS[@]}"; do
    dir="${TRAINING_DIRS[$i]}"
    echo ""
    echo "[$(( i + 1 ))/${#TRAINING_DIRS[@]}] ${dir##*/}"
    echo "------------------------------------------"
    if ! python rescore_completions.py \
      --training_dir "$dir" \
      --reward_model "$REWARD_MODEL" \
      --prefix "$PREFIX" \
      --eval_types $EVAL_TYPES \
      --num_seeds "$NUM_SEEDS" \
      --num_gpus "$NUM_GPUS" \
      --batch_size "$BATCH_SIZE" \
      --max_seq_length "$MAX_SEQ_LENGTH" \
      "${RESCORE_EVAL_OVERRIDES[@]}" \
      "${CHECKPOINT_ARGS[@]}" \
      --skip_phase2; then
      echo "WARNING: Phase 1 failed for ${dir##*/}"
      PHASE1_FAILED+=("$dir")
    fi
  done

  echo ""
  echo "---- Phase 2: Eval pipeline for complete experiments ----"
  for i in "${!TRAINING_DIRS[@]}"; do
    dir="${TRAINING_DIRS[$i]}"
    echo ""
    echo "[$(( i + 1 ))/${#TRAINING_DIRS[@]}] ${dir##*/}"
    echo "------------------------------------------"

    if ! check_phase1_complete "$dir" "$PREFIX"; then
      echo "SKIPPED: Not all checkpoints have scored data. Run Phase 1 again."
      PHASE2_SKIPPED+=("$dir")
      continue
    fi

    if ! python rescore_completions.py \
      --training_dir "$dir" \
      --reward_model "$REWARD_MODEL" \
      --prefix "$PREFIX" \
      --eval_types $EVAL_TYPES \
      --num_seeds "$NUM_SEEDS" \
      --num_gpus "$NUM_GPUS" \
      --batch_size "$BATCH_SIZE" \
      --max_seq_length "$MAX_SEQ_LENGTH" \
      "${RESCORE_EVAL_OVERRIDES[@]}" \
      "${CHECKPOINT_ARGS[@]}" \
      --skip_phase1; then
      echo "WARNING: Phase 2 failed for ${dir##*/}"
      PHASE2_FAILED+=("$dir")
    fi
  done

  TOTAL=${#TRAINING_DIRS[@]}
  P1_OK=$(( TOTAL - ${#PHASE1_FAILED[@]} ))
  P2_OK=$(( TOTAL - ${#PHASE2_FAILED[@]} - ${#PHASE2_SKIPPED[@]} ))

  echo ""
  echo "============================================================"
  echo "  [$key] Phase 1: $P1_OK/$TOTAL succeeded"
  echo "  [$key] Phase 2: $P2_OK/$TOTAL succeeded"

  if [ ${#PHASE1_FAILED[@]} -gt 0 ]; then
    echo ""
    echo "  Phase 1 FAILED (${#PHASE1_FAILED[@]}):"
    for d in "${PHASE1_FAILED[@]}"; do echo "    - $d"; done
  fi
  if [ ${#PHASE2_SKIPPED[@]} -gt 0 ]; then
    echo ""
    echo "  Phase 2 SKIPPED - incomplete Phase 1 (${#PHASE2_SKIPPED[@]}):"
    for d in "${PHASE2_SKIPPED[@]}"; do echo "    - $d"; done
  fi
  if [ ${#PHASE2_FAILED[@]} -gt 0 ]; then
    echo ""
    echo "  Phase 2 FAILED (${#PHASE2_FAILED[@]}):"
    for d in "${PHASE2_FAILED[@]}"; do echo "    - $d"; done
  fi
  echo "============================================================"

  if [ "${#CHECKPOINT_STEP_VALUES[@]}" -gt 0 ]; then
    if [ "${#PHASE1_FAILED[@]}" -gt 0 ] ||
      [ "${#PHASE2_FAILED[@]}" -gt 0 ] ||
      [ "${#PHASE2_SKIPPED[@]}" -gt 0 ]; then
      SELECTED_MODE_FAILED=1
    fi
  fi
done

if [ "$SELECTED_MODE_FAILED" -ne 0 ]; then
  exit 1
fi
