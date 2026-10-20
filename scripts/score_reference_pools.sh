#!/usr/bin/env bash
# Score reference pools with the evaluation reward models (needed for LC metrics), exactly as for the paper.
#  1) eval split of each downloaded QRPO set (50 refs/prompt) with Skywork-Llama and Skywork-Qwen
#  2) AlpacaEval reference pools (16 refs/prompt) with ArmoRM, Skywork-Llama and Skywork-Qwen
# Usage: bash scripts/score_reference_pools.sh [--use-hosted] [--dry-run]
# Skywork models are loaded by repo id; offline mode resolves the pinned revision that
# scripts/download_models.sh points the cache's main ref at.
set -euo pipefail
export HF_HUB_OFFLINE=1
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
DATA_DIR="${TUP_DATA_DIR:-$ROOT_DIR/data}"
MERGED="$DATA_DIR/shared/datasets-with-ref-rewards/merged"
ARTIFACTS="$DATA_DIR/shared/tup-eval-artifacts"
USE_HOSTED=0; DRY_RUN=0
for a in "$@"; do
  case "$a" in --use-hosted) USE_HOSTED=1 ;; --dry-run) DRY_RUN=1 ;; *) echo "ERROR: unknown option $a" >&2; exit 1 ;; esac
done
RMS=("skywork_v2_llama Skywork/Skywork-Reward-V2-Llama-3.1-8B" "skywork_v2_qwen Skywork/Skywork-Reward-V2-Qwen3-8B")

run() { printf '+'; printf ' %q' "$@"; printf '\n'; [ "$DRY_RUN" = "1" ] || "$@"; }

# Link a hosted score dir into place. Real runs refuse to create dangling links or to nest
# a link inside an existing local directory; --dry-run only prints the commands.
link_hosted() {
  local target="$1" dest="$2"
  if [ "$DRY_RUN" != "1" ]; then
    if [ ! -e "$target" ]; then
      echo "ERROR: missing hosted scores $target; fetch them with: bash scripts/download_data.sh" >&2
      exit 1
    fi
    if [ -e "$dest" ] && [ ! -L "$dest" ]; then
      echo "ERROR: $dest already exists (locally computed scores?); move it away to use the hosted copy" >&2
      exit 1
    fi
  fi
  run ln -sfn "$target" "$dest"
}

for ds in "$MERGED"/*/; do
  ds="${ds%/}"; name="$(basename "$ds")"
  case "$name" in *ultrafeedback*) split=test_prefs ;; *magpieair*) split=test ;; *) continue ;; esac
  case "$name" in mistral-nosft-*) continue ;; esac  # SFT-only set: no eval split scoring needed
  if [ "$USE_HOSTED" = "1" ]; then
    run mkdir -p "$ds/eval-split-rewards"
  fi
  for entry in "${RMS[@]}"; do
    read -r rm_name rm_path <<< "$entry"
    if [ "$USE_HOSTED" = "1" ]; then
      link_hosted "$ARTIFACTS/eval-split-rewards/$name/$rm_name" "$ds/eval-split-rewards/$rm_name"
    else
      run python score_eval_split_ref_completions.py --dataset_path "$ds" --split "$split" \
        --rm_name "$rm_name" --rm_path "$rm_path" --num_gpus 4
    fi
  done
done

for model in llama mistral-sft; do
  for entry in "armorm $DATA_DIR/shared/models/armorm" "${RMS[@]}"; do
    read -r rm_name rm_path <<< "$entry"
    if [ "$USE_HOSTED" = "1" ]; then
      run mkdir -p "$DATA_DIR/shared/datasets/alpaca-ref-completions/$model"
      link_hosted "$ARTIFACTS/alpaca-ref-completions/$model/$rm_name" "$DATA_DIR/shared/datasets/alpaca-ref-completions/$model/$rm_name"
    else
      run python generate_alpaca_ref_completions.py --skip_generation --model_key "$model" --data_dir "$DATA_DIR" \
        --rm_name "$rm_name" --rm_path "$rm_path" --num_gpus 4
    fi
  done
done
