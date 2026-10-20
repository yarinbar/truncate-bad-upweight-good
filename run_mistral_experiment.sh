#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<'EOF'
Usage:
  bash run_mistral_experiment.sh \
    --dataset-id <dataset_id> \
    --loss-type <loss_type> \
    --learning-rate <lr> \
    --beta <beta> \
    --max-steps <max_steps> \
    --scheduler-max-steps <scheduler_max_steps> \
    [--eval-steps <eval_steps>] \
    [--save-steps <save_steps>] \
    --num-ref-rewards <num_ref_rewards> \
    [--keep-all-checkpoints] \
    [--skip-eval-split] \
    [--skip-alpaca-eval] \
    [--skip-rescore] \
    [--eval-base-only] \
    [--model-path <model_path>] \
    [--ref-model-key <ref_model_key>] \
    [--run-name <run_name>] \
    [--qrpo-transform-type <transform>] \
    [--qrpo-loss-type <qrpo_loss_type>] \
    [--qrpo-sample-selector <selector>] \
    [--binomial-lambda <binomial_lambda>] \
    [--use-discrete-partition <true_or_false>] \
    [--use-length-normalized-logratios <true_or_false>]

Note: --keep-all-checkpoints evaluates every kept training checkpoint back to back, one vLLM pool each; a run may need to be resumed (rerun the same command) if GPU memory is slow to free.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" > /dev/null 2>&1; then
    die "Required command not found on PATH: $cmd"
  fi
}

build_auto_job_name() {
  local loss_name="${LOSS_TYPE//\//-}"
  local job_name="${DATASET_ID}-${loss_name}-${QRPO_TRANSFORM_TYPE}-applytempFalse-numref${NUM_REF_REWARDS}-lr${LEARNING_RATE}-beta${BETA}"
  local sample_selector="${QRPO_SAMPLE_SELECTOR:-both}"

  if [ -n "$sample_selector" ]; then
    job_name="${job_name}-sampleSel${sample_selector}"
  fi

  if [ -n "$BINOMIAL_LAMBDA" ]; then
    job_name="${job_name}-binomlam${BINOMIAL_LAMBDA}"
  fi

  if [ -n "$USE_DISCRETE_PARTITION" ]; then
    job_name="${job_name}-discretepart${USE_DISCRETE_PARTITION}"
  fi

  if [ -n "$USE_LENGTH_NORMALIZED_LOGRATIOS" ]; then
    job_name="${job_name}-lennorm${USE_LENGTH_NORMALIZED_LOGRATIOS}"
  fi

  if [ -n "$SCHEDULER_MAX_STEPS" ]; then
    job_name="${job_name}-schedsteps${SCHEDULER_MAX_STEPS}"
  fi

  if [ -n "$EVAL_STEPS" ]; then
    job_name="${job_name}-evalsteps${EVAL_STEPS}"
  fi

  if [ -n "$SAVE_STEPS" ]; then
    job_name="${job_name}-savesteps${SAVE_STEPS}"
  fi

  printf '%s\n' "$job_name"
}

resolve_dataset_name() {
  case "$1" in
    *ultrafeedback*)
      printf '%s\n' "ultrafeedback-armorm"
      ;;
    *magpieair*)
      printf '%s\n' "magpieair-armorm"
      ;;
    *)
      die "Unsupported dataset family in --dataset-id: $1"
      ;;
  esac
}

default_save_steps_for_dataset_id() {
  case "$1" in
    *ultrafeedback*)
      printf '%s\n' "100"
      ;;
    *magpieair*)
      printf '%s\n' "160"
      ;;
    *)
      die "Unsupported dataset family in --dataset-id: $1"
      ;;
  esac
}

print_cmd() {
  printf '+'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

require_dir() {
  local path="$1"
  local label="$2"
  if [ ! -d "$path" ]; then
    die "$label not found: $path"
  fi
}

require_python_modules() {
  local missing
  if ! missing=$(
    python - "$@" <<'PY'
import importlib
import sys

missing = []
for module_name in sys.argv[1:]:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        missing.append(f"{module_name}: {exc.__class__.__name__}: {exc}")

if missing:
    print("\n".join(missing))
    sys.exit(1)
PY
  ); then
    die "Python environment is missing required modules:
$missing"
  fi
}

require_visible_gpus() {
  local min_count="$1"
  local gpu_count
  if ! gpu_count=$(
    python - "$min_count" <<'PY'
import sys
import torch

required = int(sys.argv[1])
count = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(count)
sys.exit(0 if count >= required else 1)
PY
  ); then
    die "Expected at least $min_count visible CUDA GPUs, found ${gpu_count:-0}"
  fi
}

require_hf_model_cached() {
  local repo_id="$1"
  local resolved_path
  if ! resolved_path=$(
    python - "$repo_id" <<'PY'
import sys

from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
print(snapshot_download(repo_id=repo_id, local_files_only=True))
PY
  ); then
    die "Required Hugging Face model is not available in local cache: $repo_id"
  fi
  echo "Found cached Hugging Face model: $repo_id -> $resolved_path"
}

report_eval_split_reward_status() {
  local dataset_path="$1"
  local rm_name="$2"
  local reward_dir="$dataset_path/eval-split-rewards/$rm_name"
  if [ -d "$reward_dir" ]; then
    echo "Found dataset-local eval-split rewards: $reward_dir"
  else
    echo "Dataset-local eval-split rewards will be created during rescoring: $reward_dir"
  fi
}

require_single_hash_dir() {
  local checkpoints_root="$1"
  mapfile -t HASH_DIRS < <(find "$checkpoints_root" -mindepth 1 -maxdepth 1 -type d | sort)
  if [ "${#HASH_DIRS[@]}" -ne 1 ]; then
    die "Expected exactly one hash directory under $checkpoints_root, found ${#HASH_DIRS[@]}"
  fi
  HASH_DIR="${HASH_DIRS[0]}"
}

require_checkpoint_dirs() {
  local hash_dir="$1"
  if ! compgen -G "$hash_dir/checkpoint-*" > /dev/null; then
    die "No checkpoint directories found under $hash_dir"
  fi
}

prune_intermediate_checkpoints() {
  local hash_dir="$1"
  local checkpoint_step="$2"
  local keep_checkpoint_dir="$hash_dir/checkpoint-$checkpoint_step"
  local checkpoint_dir
  local pruned_count=0

  if [ "$KEEP_ALL_CHECKPOINTS" = "1" ]; then
    echo "== Keeping all checkpoints for $RUN_LABEL =="
    return
  fi

  if [ ! -d "$keep_checkpoint_dir" ]; then
    die "Cannot prune checkpoints because final checkpoint is missing: $keep_checkpoint_dir"
  fi

  shopt -s nullglob
  for checkpoint_dir in "$hash_dir"/checkpoint-*; do
    [ -d "$checkpoint_dir" ] || continue
    if [ "$checkpoint_dir" = "$keep_checkpoint_dir" ]; then
      continue
    fi
    rm -rf "$checkpoint_dir"
    pruned_count=$((pruned_count + 1))
  done
  shopt -u nullglob

  echo "== Pruned $pruned_count checkpoint directories; kept $keep_checkpoint_dir =="
}

training_checkpoint_exists() {
  local hash_dir="$1"
  local checkpoint_step="$2"
  [ -d "$hash_dir/checkpoint-$checkpoint_step" ]
}

kept_training_steps() {
  local hash_dir="$1"
  local checkpoint_dir
  local step
  local -a steps=()

  # Every kept checkpoint-<N> directory with N > 0, ascending, space separated. checkpoint-0 (the base model) and
  # symlinks (run_online_eval.py links a missing checkpoint-0 to the base-model directory) are not training steps.
  for checkpoint_dir in "$hash_dir"/checkpoint-*; do
    if [ ! -d "$checkpoint_dir" ] || [ -L "$checkpoint_dir" ]; then
      continue
    fi
    step="${checkpoint_dir##*/checkpoint-}"
    if [[ "$step" =~ ^[1-9][0-9]*$ ]]; then
      steps+=("$step")
    fi
  done
  if [ "${#steps[@]}" -eq 0 ]; then
    die "No training checkpoints (step > 0) found under $hash_dir"
  fi
  printf '%s\n' "${steps[@]}" | sort -n | tr '\n' ' ' | sed 's/ $//'
}

prepare_base_checkpoint_dir() {
  local hash_dir="$1"
  local model_path="$2"
  local checkpoint_dir="$hash_dir/checkpoint-0"
  local model_abs
  local src
  local name

  # run_online_eval.py symlinks a missing checkpoint-0 to the shared base-model directory, which
  # would write this run's eval outputs into that directory and reuse another setting's outputs.
  # Give every base-only run its own real checkpoint-0 that links only the model files.
  if [ -L "$checkpoint_dir" ]; then
    die "Refusing to reuse symlinked base checkpoint (eval outputs would go into the shared model directory): $checkpoint_dir"
  fi

  model_abs="$(cd "$model_path" && pwd -P)"
  mkdir -p "$checkpoint_dir"
  for src in "$model_abs"/*; do
    [ -f "$src" ] || continue
    name="$(basename "$src")"
    case "$name" in
      *online_evals*|*alpaca_eval*|*leaderboard*)
        continue
        ;;
    esac
    if [ ! -e "$checkpoint_dir/$name" ] && [ ! -L "$checkpoint_dir/$name" ]; then
      ln -s "$src" "$checkpoint_dir/$name"
    fi
  done
}

write_base_training_config() {
  local config_path="$1"
  shift

  # rescore_completions.py discovers runs through checkpoints/*/config/process-0/config-postprocessed.yaml
  # and builds its eval commands from that saved training config, so a base-only run needs one too.
  # Compose the config this run's training command would have used (same Hydra overrides).
  if [ -f "$config_path" ]; then
    return
  fi

  python - "$ROOT_DIR" "$config_path" "$@" <<'PY'
import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import qrpo.utils.config as qrpo_config

root_dir, config_path, *train_cmd = sys.argv[1:]
overrides = train_cmd[train_cmd.index("qrpo.train_qrpo") + 1 :]
overrides.append(f"data_dir={root_dir}/data")

qrpo_config.register_resolvers()
with initialize_config_dir(
    config_dir=str(Path(root_dir) / "src" / "qrpo" / "configs"),
    version_base=None,
    job_name="train_qrpo",
):
    cfg = compose(config_name="train-qrpo", overrides=overrides, return_hydra_config=True)
    HydraConfig.instance().set_config(cfg)
    # Resolve every key except the Hydra runtime node itself (parts of it, e.g. hydra.job.num,
    # only exist inside a running Hydra app and are not part of a training config).
    resolved = {}
    for key in cfg.keys():
        if key == "hydra":
            continue
        node = OmegaConf.select(cfg, key)
        resolved[key] = OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else node
    text = OmegaConf.to_yaml(OmegaConf.create(resolved))

path = Path(config_path)
path.parent.mkdir(parents=True, exist_ok=True)
tmp_path = path.with_name(path.name + ".tmp")
tmp_path.write_text(
    "# Composed by the launcher for --eval-base-only (no training ran); same overrides as the training command.\n"
    + text
)
os.replace(tmp_path, path)
PY
}

require_eval_outputs() {
  local hash_dir="$1"
  local eval_type="$2"
  local checkpoint_step="${3:-0}"
  local required_path="$hash_dir/checkpoint-$checkpoint_step/online_evals_0/$eval_type/kl-dataset"
  if [ ! -d "$required_path" ]; then
    die "Missing eval output: $required_path"
  fi
}

require_rescore_outputs() {
  local hash_dir="$1"
  local prefix="$2"
  local dataset_path="$3"
  local rm_name="$4"
  local checkpoint_step="${5:-}"
  local reward_dir="$dataset_path/eval-split-rewards/$rm_name"

  if [ ! -d "$reward_dir" ]; then
    die "Missing dataset-local rescored ref rewards: $reward_dir"
  fi

  if [ -n "$checkpoint_step" ]; then
    local required_path="$hash_dir/checkpoint-$checkpoint_step/${prefix}online_evals_0"
    if [ ! -d "$required_path" ]; then
      die "Missing rescored online eval output: $required_path"
    fi
    local eval_type
    local completion_log
    for eval_type in "${ENABLED_RESCORING_EVAL_TYPES[@]}"; do
      completion_log="$hash_dir/checkpoint-$checkpoint_step/${prefix}online_evals_logs_${eval_type}.json"
      if [ ! -f "$completion_log" ]; then
        die "Missing rescored online eval completion log: $completion_log"
      fi
    done
  else
    if ! find "$hash_dir" -type d -name "${prefix}online_evals_0" | grep -q .; then
      die "Missing rescored online eval outputs for prefix ${prefix} under $hash_dir"
    fi
  fi
}

DATASET_ID=""
LOSS_TYPE=""
LEARNING_RATE=""
BETA=""
MAX_STEPS=""
SCHEDULER_MAX_STEPS=""
EVAL_STEPS=""
SAVE_STEPS=""
NUM_REF_REWARDS=""
MODEL_PATH_OVERRIDE=""
REF_MODEL_KEY="mistral"
RUN_NAME=""
QRPO_TRANSFORM_TYPE="identity"
QRPO_LOSS_TYPE="mse"
QRPO_SAMPLE_SELECTOR="both"
BINOMIAL_LAMBDA=""
USE_DISCRETE_PARTITION=""
USE_LENGTH_NORMALIZED_LOGRATIOS=""
KEEP_ALL_CHECKPOINTS="0"
SKIP_EVAL_SPLIT="0"
SKIP_ALPACA_EVAL="0"
SKIP_RESCORE="0"
EVAL_BASE_ONLY="0"
DRY_RUN="${DRY_RUN:-0}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset-id)
      DATASET_ID="$2"
      shift 2
      ;;
    --loss-type)
      LOSS_TYPE="$2"
      shift 2
      ;;
    --learning-rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --beta)
      BETA="$2"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --scheduler-max-steps)
      SCHEDULER_MAX_STEPS="$2"
      shift 2
      ;;
    --eval-steps)
      EVAL_STEPS="$2"
      shift 2
      ;;
    --save-steps)
      SAVE_STEPS="$2"
      shift 2
      ;;
    --num-ref-rewards)
      NUM_REF_REWARDS="$2"
      shift 2
      ;;
    --keep-all-checkpoints)
      KEEP_ALL_CHECKPOINTS="1"
      shift
      ;;
    --skip-eval-split)
      SKIP_EVAL_SPLIT="1"
      shift
      ;;
    --skip-alpaca-eval)
      SKIP_ALPACA_EVAL="1"
      shift
      ;;
    --skip-rescore)
      SKIP_RESCORE="1"
      shift
      ;;
    --model-path)
      MODEL_PATH_OVERRIDE="$2"
      shift 2
      ;;
    --ref-model-key)
      REF_MODEL_KEY="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --qrpo-transform-type)
      QRPO_TRANSFORM_TYPE="$2"
      shift 2
      ;;
    --qrpo-loss-type)
      QRPO_LOSS_TYPE="$2"
      shift 2
      ;;
    --qrpo-sample-selector)
      QRPO_SAMPLE_SELECTOR="$2"
      shift 2
      ;;
    --binomial-lambda)
      BINOMIAL_LAMBDA="$2"
      shift 2
      ;;
    --use-discrete-partition)
      USE_DISCRETE_PARTITION="$2"
      shift 2
      ;;
    --use-length-normalized-logratios)
      USE_LENGTH_NORMALIZED_LOGRATIOS="$2"
      shift 2
      ;;
    --eval-base-only)
      EVAL_BASE_ONLY="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      die "Unknown argument: $1"
      ;;
  esac
done

[ -n "$DATASET_ID" ] || die "--dataset-id is required"
[ -n "$LOSS_TYPE" ] || die "--loss-type is required"
[ -n "$LEARNING_RATE" ] || die "--learning-rate is required"
[ -n "$BETA" ] || die "--beta is required"
[ -n "$MAX_STEPS" ] || die "--max-steps is required"
[ -n "$NUM_REF_REWARDS" ] || die "--num-ref-rewards is required"
[ -n "$SCHEDULER_MAX_STEPS" ] || die "--scheduler-max-steps is required"

if [ -n "$SCHEDULER_MAX_STEPS" ] && [ "$SCHEDULER_MAX_STEPS" -lt "$MAX_STEPS" ]; then
  die "--scheduler-max-steps must be greater than or equal to --max-steps"
fi

# Checkpoints covered by the eval and rescoring stages (EVAL_CHECKPOINT_STEPS, space separated). A trained run never
# evaluates checkpoint-0: it evaluates and rescores only its stop checkpoint (checkpoint-$MAX_STEPS), the one the
# paper reports. Also evaluating checkpoint-0 started a second vLLM round right after the first, while the previous
# workers still held GPU memory (CUDA OOM). The base model's numbers are the `initial` rows (--eval-base-only),
# which evaluate only their own checkpoint-0 and leave this empty.
# With --keep-all-checkpoints it is every kept training step, set after the training stage.
EVAL_CHECKPOINT_STEPS=""
if [ "$EVAL_BASE_ONLY" != "1" ] && [ "$KEEP_ALL_CHECKPOINTS" != "1" ]; then
  EVAL_CHECKPOINT_STEPS="$MAX_STEPS"
fi

MODEL_NAME="mistral"
DATASET_NAME="$(resolve_dataset_name "$DATASET_ID")"
MODEL_PATH_DEFAULT="$ROOT_DIR/data/shared/models/mistral"
MODEL_PATH="${MODEL_PATH_OVERRIDE:-$MODEL_PATH_DEFAULT}"
DATASET_PATH="$ROOT_DIR/data/shared/datasets-with-ref-rewards/merged/$DATASET_ID"
ALPACA_DATASET_PATH="$ROOT_DIR/data/shared/datasets/alpaca"
RUN_PREFIX="chat-baseline"
if [ "$EVAL_BASE_ONLY" = "1" ] && [ -z "$RUN_NAME" ]; then
  RUN_NAME="initial-${DATASET_ID}"
fi
JOB_NAME="$RUN_NAME"
if [ -z "$JOB_NAME" ]; then
  JOB_NAME="$(build_auto_job_name)"
fi
JOB_SUBDIR="$RUN_PREFIX/$JOB_NAME"
RUN_ROOT="$ROOT_DIR/outputs/shared/train_qrpo/$JOB_SUBDIR"
CHECKPOINTS_ROOT="$RUN_ROOT/checkpoints"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
RUN_LABEL="$JOB_SUBDIR"
CHECK_ONLY="${CHECK_ONLY:-0}"
SAVE_STEPS_TO_USE="${SAVE_STEPS:-$(default_save_steps_for_dataset_id "$DATASET_ID")}"

if [ "$DRY_RUN" != "1" ]; then
  require_command python
  require_command accelerate
  require_python_modules qrpo accelerate datasets torch transformers wandb trl hydra omegaconf huggingface_hub yaml
  require_visible_gpus "$NPROC_PER_NODE"
  require_dir "$MODEL_PATH" "Model path"
  require_dir "$DATASET_PATH" "Dataset path"
  if [ "$SKIP_ALPACA_EVAL" != "1" ]; then
    require_dir "$ALPACA_DATASET_PATH" "Alpaca eval dataset path"
  fi
  if [ "$SKIP_RESCORE" != "1" ]; then
    require_hf_model_cached "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
    require_hf_model_cached "Skywork/Skywork-Reward-V2-Qwen3-8B"
    report_eval_split_reward_status "$DATASET_PATH" "skywork_v2_llama"
    report_eval_split_reward_status "$DATASET_PATH" "skywork_v2_qwen"
  fi
fi

if [ "$CHECK_ONLY" = "1" ]; then
  echo "== Preflight OK: $RUN_LABEL =="
  exit 0
fi

TRAIN_CMD=(
  accelerate launch
  --config-file src/qrpo/configs/accelerate/ds-zero1.yaml
  --num_machines 1
  --num_processes "$NPROC_PER_NODE"
  --main_process_port "$MASTER_PORT"
  -m qrpo.train_qrpo
  "dataset=$DATASET_NAME"
  "dataset_args.dataset_path=$DATASET_PATH"
  "model=$MODEL_NAME"
  "model_args.model_name_or_path=$MODEL_PATH"
  "model_generation_config.temperature=1.0"
  "training_args.max_grad_norm=1e8"
  "training_args.gradient_accumulation_steps=16"
  "training_args.per_device_train_batch_size=2"
  "training_args.per_device_eval_batch_size=8"
  "training_args.learning_rate=$LEARNING_RATE"
  "training_args.num_train_epochs=$NUM_TRAIN_EPOCHS"
  "training_args.loss_type=$LOSS_TYPE"
  "training_args.qrpo_transform_type=$QRPO_TRANSFORM_TYPE"
  "training_args.qrpo_loss_type=$QRPO_LOSS_TYPE"
  "training_args.qrpo_sample_selector=$QRPO_SAMPLE_SELECTOR"
  "training_args.apply_temperature_in_training=False"
  "training_args.num_ref_rewards=$NUM_REF_REWARDS"
  "training_args.beta=$BETA"
  "training_args.bonbon_alpha=0.0"
  "job_subdir=$JOB_SUBDIR"
  "outputs_subdir=shared"
  "resuming.resume=True"
  "++ref_model_key=$REF_MODEL_KEY"
)

  TRAIN_CMD+=("stop_after_steps=$MAX_STEPS")
TRAIN_CMD+=("wandb.run_name=${LOSS_TYPE}-${JOB_NAME}-${RUN_PREFIX}")

if [ -n "$BINOMIAL_LAMBDA" ]; then
  TRAIN_CMD+=("training_args.binomial_lambda=$BINOMIAL_LAMBDA")
fi

if [ -n "$USE_DISCRETE_PARTITION" ]; then
  TRAIN_CMD+=("training_args.use_discrete_partition=$USE_DISCRETE_PARTITION")
fi

if [ -n "$USE_LENGTH_NORMALIZED_LOGRATIOS" ]; then
  TRAIN_CMD+=("training_args.use_length_normalized_logratios=$USE_LENGTH_NORMALIZED_LOGRATIOS")
fi

if [ -n "$SCHEDULER_MAX_STEPS" ]; then
  TRAIN_CMD+=("training_args.max_steps=$SCHEDULER_MAX_STEPS")
fi

if [ -n "$EVAL_STEPS" ]; then
  TRAIN_CMD+=("training_args.eval_steps=$EVAL_STEPS")
fi

  TRAIN_CMD+=("training_args.save_steps=$SAVE_STEPS_TO_USE")

if [ "$DRY_RUN" = "1" ]; then
  if [ "$EVAL_BASE_ONLY" = "1" ]; then
    printf '+ eval-base-only %s\n' "$RUN_ROOT"
  else
    print_cmd "${TRAIN_CMD[@]}"
  fi
  echo "RUN_ROOT=$RUN_ROOT"
  exit 0
fi

export HYDRA_FULL_ERROR=1
if [ -e "$RUN_ROOT" ] && [ ! -d "$RUN_ROOT" ]; then
  die "Run root exists but is not a directory: $RUN_ROOT"
fi

if [ "$EVAL_BASE_ONLY" = "1" ]; then
  mkdir -p "$CHECKPOINTS_ROOT/base"
  require_single_hash_dir "$CHECKPOINTS_ROOT"
  prepare_base_checkpoint_dir "$HASH_DIR" "$MODEL_PATH"
  write_base_training_config "$HASH_DIR/config/process-0/config-postprocessed.yaml" "${TRAIN_CMD[@]}"
  echo "== Evaluating base model only: $RUN_LABEL =="
else
  if [ -d "$CHECKPOINTS_ROOT" ]; then
    require_single_hash_dir "$CHECKPOINTS_ROOT"
    if training_checkpoint_exists "$HASH_DIR" "$MAX_STEPS"; then
      echo "== Training already completed at $HASH_DIR/checkpoint-$MAX_STEPS; skipping training for $RUN_LABEL =="
    else
      echo "== Resuming training: $RUN_LABEL =="
      print_cmd "${TRAIN_CMD[@]}"
      "${TRAIN_CMD[@]}"
    fi
  else
    echo "== Training: $RUN_LABEL =="
    print_cmd "${TRAIN_CMD[@]}"
    "${TRAIN_CMD[@]}"
  fi

  require_dir "$CHECKPOINTS_ROOT" "Checkpoints root"
  require_single_hash_dir "$CHECKPOINTS_ROOT"
  require_checkpoint_dirs "$HASH_DIR"
  prune_intermediate_checkpoints "$HASH_DIR" "$MAX_STEPS"
  if [ "$KEEP_ALL_CHECKPOINTS" = "1" ] && [ -z "$EVAL_CHECKPOINT_STEPS" ]; then
    training_checkpoint_exists "$HASH_DIR" "$MAX_STEPS" \
      || die "Cannot evaluate kept checkpoints because final checkpoint is missing: $HASH_DIR/checkpoint-$MAX_STEPS"
    EVAL_CHECKPOINT_STEPS="$(kept_training_steps "$HASH_DIR")"
    echo "== Evaluating kept checkpoints (steps $EVAL_CHECKPOINT_STEPS): $RUN_LABEL =="
  fi
fi

run_eval() {
  local eval_type="$1"
  local -a eval_cmd=(
    python -m qrpo.evals.run_online_eval
    "dataset=$DATASET_NAME"
    "dataset_args.dataset_path=$DATASET_PATH"
    "model=$MODEL_NAME"
    "model_args.model_name_or_path=$MODEL_PATH"
    "training_dir=$RUN_ROOT"
    "eval_type=$eval_type"
    "custom_eval_prefix="
    "training_args.loss_type=$LOSS_TYPE"
    "training_args.qrpo_transform_type=$QRPO_TRANSFORM_TYPE"
    "training_args.qrpo_loss_type=$QRPO_LOSS_TYPE"
    "training_args.num_ref_rewards=$NUM_REF_REWARDS"
    "training_args.beta=$BETA"
    "job_subdir=$JOB_SUBDIR"
    "outputs_subdir=shared"
    "resuming.resume=False"
    "wandb.mode=disabled"
    "++ref_model_key=$REF_MODEL_KEY"
  )

  if [ -n "$EVAL_CHECKPOINT_STEPS" ]; then
    eval_cmd+=("checkpoint_steps=[${EVAL_CHECKPOINT_STEPS// /,}]")
  fi

  echo "== Online eval ($eval_type): $RUN_LABEL =="
  print_cmd "${eval_cmd[@]}"
  "${eval_cmd[@]}"
}

ENABLED_RESCORING_EVAL_TYPES=()

if [ "$SKIP_EVAL_SPLIT" != "1" ]; then
  run_eval "eval_split"
  ENABLED_RESCORING_EVAL_TYPES+=("eval_split")
else
  echo "== Skipping online eval (eval_split): $RUN_LABEL =="
fi

if [ "$SKIP_ALPACA_EVAL" != "1" ]; then
  run_eval "alpaca_eval"
  ENABLED_RESCORING_EVAL_TYPES+=("alpaca_eval")
else
  echo "== Skipping online eval (alpaca_eval): $RUN_LABEL =="
fi

require_single_hash_dir "$CHECKPOINTS_ROOT"
for eval_output_step in ${EVAL_CHECKPOINT_STEPS:-0}; do
  if [ "$SKIP_EVAL_SPLIT" != "1" ]; then
    require_eval_outputs "$HASH_DIR" "eval_split" "$eval_output_step"
  fi
  if [ "$SKIP_ALPACA_EVAL" != "1" ]; then
    require_eval_outputs "$HASH_DIR" "alpaca_eval" "$eval_output_step"
  fi
done

TMP_DIRS_FILE=""
cleanup() {
  if [ -n "$TMP_DIRS_FILE" ]; then
    rm -f "$TMP_DIRS_FILE"
  fi
}
trap cleanup EXIT

if [ "$SKIP_RESCORE" != "1" ]; then
  if [ "${#ENABLED_RESCORING_EVAL_TYPES[@]}" -eq 0 ]; then
    echo "== Skipping rescoring because all baseline evals were skipped: $RUN_LABEL =="
  else
    TMP_DIRS_FILE="$(mktemp)"
    printf '%s\n' "$RUN_ROOT" > "$TMP_DIRS_FILE"

    echo "== Rescoring: $RUN_LABEL =="
    print_cmd env "DIRS_FILE=$TMP_DIRS_FILE" "EVAL_TYPES=${ENABLED_RESCORING_EVAL_TYPES[*]}" "NUM_GPUS=$NPROC_PER_NODE" "RESCORE_MODEL=$MODEL_NAME" "RESCORE_DATASET=$DATASET_NAME" "RESCORE_REF_MODEL_KEY=$REF_MODEL_KEY" "CHECKPOINT_STEPS=$EVAL_CHECKPOINT_STEPS" bash "$ROOT_DIR/run_rescore_multi_rm.sh" skywork_v2_llama skywork_v2_qwen
    DIRS_FILE="$TMP_DIRS_FILE" EVAL_TYPES="${ENABLED_RESCORING_EVAL_TYPES[*]}" NUM_GPUS="$NPROC_PER_NODE" RESCORE_MODEL="$MODEL_NAME" RESCORE_DATASET="$DATASET_NAME" RESCORE_REF_MODEL_KEY="$REF_MODEL_KEY" CHECKPOINT_STEPS="$EVAL_CHECKPOINT_STEPS" bash "$ROOT_DIR/run_rescore_multi_rm.sh" skywork_v2_llama skywork_v2_qwen

    require_single_hash_dir "$CHECKPOINTS_ROOT"
    for rm_name in skywork_v2_llama skywork_v2_qwen; do
      if [ -z "$EVAL_CHECKPOINT_STEPS" ]; then
        require_rescore_outputs "$HASH_DIR" "${rm_name}_" "$DATASET_PATH" "$rm_name"
      else
        for rescore_step in $EVAL_CHECKPOINT_STEPS; do
          require_rescore_outputs "$HASH_DIR" "${rm_name}_" "$DATASET_PATH" "$rm_name" "$rescore_step"
        done
      fi
    done
  fi
else
  echo "== Skipping rescoring: $RUN_LABEL =="
fi

echo "== Completed: $RUN_LABEL =="
