#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# run_alpaca_eval_api.sh — Run AlpacaEval API scoring on selected checkpoints.
#
# This script scores already-generated AlpacaEval completions from:
#   <checkpoint>/online_evals_0/alpaca_eval/model_outputs.json
#
# Usage:
#   ./run_alpaca_eval_api.sh --list
#   ./run_alpaca_eval_api.sh --list --search-root outputs/saved/llama_ultrafeedback
#   ./run_alpaca_eval_api.sh --annotators-configs "alpaca_eval_gpt4_turbo_fn alpaca_eval_gpt4o_fn" \
#       --experiment outputs/saved/llama_ultrafeedback/some-run --last-only
#   ./run_alpaca_eval_api.sh --annotators-configs "alpaca_eval_gpt4_turbo_fn" \
#       --from-file checkpoints.txt
# ==============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

SEARCH_ROOT="${ROOT_DIR}/outputs/shared/train_qrpo"
LOG_DIR="${ROOT_DIR}/run_logs/alpaca_eval_api"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

MODE="run"
DRY_RUN=false
FILTER=""
STEP=""
LAST_ONLY=false
FROM_FILE=""
SEARCH_ROOT_SET=false
WANDB_PROJECT=""
JOB_SUBDIR_PREFIX="alpaca-eval-api"
ANNOTATORS_CONFIGS=()
CHECKPOINT_ARGS=()
EXPERIMENT_ARGS=()

usage() {
  sed -n '3,/^# ==/p' "$0" | head -n -1 | sed 's/^# \?//'
  exit 0
}

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_section() {
  echo -e "\n${BLUE}============================================================${NC}"
  echo -e "${BLUE}$*${NC}"
  echo -e "${BLUE}============================================================${NC}"
}

trim() {
  echo "$1" | xargs
}

normalize_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    echo "$path"
  else
    echo "${ROOT_DIR}/$path"
  fi
}

dedupe_preserving_order() {
  declare -A seen=()
  local item=""
  for item in "$@"; do
    if [[ -n "${seen["$item"]+x}" ]]; then
      continue
    fi
    seen["$item"]=1
    printf '%s\n' "$item"
  done
}

require_command() {
  local name="$1"
  command -v "$name" >/dev/null 2>&1 || {
    log_error "Missing required command: $name"
    exit 1
  }
}

require_python_runtime() {
  local output=""
  if ! output="$(
    CODEX_ROOT_DIR="$ROOT_DIR" python - <<'PY' 2>&1
import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.environ["CODEX_ROOT_DIR"], "src"))

required = {
    "qrpo": "repo package (qrpo, via $ROOT_DIR/src)",
    "hydra": "hydra-core",
    "omegaconf": "omegaconf",
    "pandas": "pandas",
    "wandb": "wandb",
}

missing = [label for module, label in required.items() if importlib.util.find_spec(module) is None]
if missing:
    raise SystemExit("Missing Python packages: " + ", ".join(missing))
PY
  )"; then
    log_error "${output}"
    exit 1
  fi
}

require_runtime_dependencies() {
  require_command python
  require_command alpaca_eval
  require_command sha256sum
  require_python_runtime
}

parse_annotators_configs() {
  local raw="$1"
  local parsed=()
  # shellcheck disable=SC2206
  parsed=($raw)
  if [[ ${#parsed[@]} -eq 0 ]]; then
    log_error "--annotators-configs requires at least one config"
    exit 1
  fi
  ANNOTATORS_CONFIGS=("${parsed[@]}")
}

infer_judge_model() {
  local judge_config="$1"
  local resolved=""

  if resolved="$(
    CODEX_ROOT_DIR="$ROOT_DIR" CODEX_JUDGE_CONFIG="$judge_config" python - <<'PY' 2>/dev/null
import os
import sys

sys.path.insert(0, os.path.join(os.environ["CODEX_ROOT_DIR"], "src"))
from qrpo.evals.judge_metadata import infer_judge_model

print(infer_judge_model(os.environ["CODEX_JUDGE_CONFIG"]))
PY
  )"; then
    if [[ -n "${resolved}" ]]; then
      echo "${resolved}"
      return
    fi
  fi

  local lowered="${judge_config,,}"
  case "${lowered}" in
    *gpt4o*mini*) echo "gpt-4o-mini" ;;
    *gpt4o*) echo "gpt-4o" ;;
    *gpt4*turbo*|*gpt4_turbo*) echo "gpt-4-turbo" ;;
    *gpt4_1*mini*|*gpt41*mini*) echo "gpt-4.1-mini" ;;
    *gpt4_1*nano*|*gpt41*nano*) echo "gpt-4.1-nano" ;;
    *gpt4_1*|*gpt41*) echo "gpt-4.1" ;;
    *) echo "$judge_config" ;;
  esac
}

sanitize_job_component() {
  local raw="$1"
  local sanitized=""
  sanitized="$(printf '%s' "$raw" | tr -cs '[:alnum:]_.=-' '-')"
  sanitized="${sanitized#-}"
  sanitized="${sanitized%-}"
  if [[ -z "${sanitized}" ]]; then
    echo "item"
  else
    echo "${sanitized}"
  fi
}

short_path_hash() {
  printf '%s' "$1" | sha256sum | cut -c1-12
}

make_job_id() {
  local checkpoint_path="${1%/}"
  local checkpoint_name="${checkpoint_path##*/}"
  local hash_path="${checkpoint_path%/*}"
  local checkpoint_hash="${hash_path##*/}"
  local checkpoints_dir="${hash_path%/*}"
  local experiment_dir="${checkpoints_dir%/*}"
  local experiment_name="${experiment_dir##*/}"
  local experiment_slug=""
  local checkpoint_hash_slug=""
  local checkpoint_slug=""
  local path_hash=""

  experiment_slug="$(sanitize_job_component "${experiment_name}" | cut -c1-80)"
  checkpoint_hash_slug="$(sanitize_job_component "${checkpoint_hash}" | cut -c1-32)"
  checkpoint_slug="$(sanitize_job_component "${checkpoint_name}" | cut -c1-32)"
  path_hash="$(short_path_hash "${checkpoint_path}")"

  printf '%s-%s-%s-%s\n' \
    "${experiment_slug}" \
    "${checkpoint_hash_slug}" \
    "${checkpoint_slug}" \
    "${path_hash}"
}

has_completions() {
  local checkpoint_path="$1"
  [[ -f "${checkpoint_path}/online_evals_0/alpaca_eval/model_outputs.json" ]]
}

has_results_for_judge() {
  local checkpoint_path="$1"
  local judge="$2"

  if [[ -d "${checkpoint_path}/${judge}" ]]; then
    return 0
  fi

  if [[ ! -f "${checkpoint_path}/leaderboard.csv" || ! -f "${checkpoint_path}/annotations.json" ]]; then
    return 1
  fi

  CODEX_CHECKPOINT_PATH="$checkpoint_path" CODEX_JUDGE_CONFIG="$judge" python - <<'PY' >/dev/null 2>&1
import json
import os
import sys
from pathlib import Path

checkpoint_path = Path(os.environ["CODEX_CHECKPOINT_PATH"])
judge_config = os.environ["CODEX_JUDGE_CONFIG"]
annotations_file = checkpoint_path / "annotations.json"

try:
    with open(annotations_file, "r") as f:
        annotations = json.load(f)
except (OSError, json.JSONDecodeError, TypeError):
    sys.exit(1)

annotators = {
    str(annotation.get("annotator")).strip()
    for annotation in annotations
    if isinstance(annotation, dict) and annotation.get("annotator")
}
sys.exit(0 if annotators == {judge_config} else 1)
PY
}

has_results_for_all_judges() {
  local checkpoint_path="$1"
  local judge
  for judge in "${ANNOTATORS_CONFIGS[@]}"; do
    has_results_for_judge "${checkpoint_path}" "${judge}" || return 1
  done
  return 0
}

experiment_has_checkpoints() {
  local experiment_dir="$1"
  find "${experiment_dir}" -mindepth 2 -maxdepth 3 -type d -name "checkpoint-*" | grep -q .
}

find_experiments() {
  local filter="${1:-}"
  local candidate=""
  local priority=""
  find "${SEARCH_ROOT}" -type d | while read -r candidate; do
    if ! [[ -d "${candidate}" ]]; then
      continue
    fi
    if [[ "$(basename "${candidate}")" == "checkpoints" ]]; then
      continue
    fi
    if ! experiment_has_checkpoints "${candidate}"; then
      continue
    fi
    if [[ -n "${filter}" && "${candidate}" != *"${filter}"* ]]; then
      continue
    fi
    priority=9
    case "${candidate,,}" in
      *-armorm-ours-*)
        priority=0
        ;;
      *-armorm-simpo-*)
        priority=1
        ;;
      *offpolicy2best*-armorm-rebel-*)
        priority=2
        ;;
    esac
    printf '%s\t%s\n' "${priority}" "${candidate}"
  done | sort -t $'\t' -k1,1n -k2,2 | cut -f2- | uniq
}

find_checkpoints() {
  local experiment_dir="$1"
  local step_filter="${2:-}"
  local pattern="checkpoint-*"
  if [[ -n "${step_filter}" ]]; then
    pattern="checkpoint-${step_filter}"
  fi
  find "${experiment_dir}" \
    -mindepth 2 \
    -maxdepth 3 \
    -type d \
    -name "${pattern}" \
    ! -name "checkpoint-0" | sort -u | sort -t'-' -k2 -n
}

find_last_checkpoint() {
  local experiment_dir="$1"
  find_checkpoints "${experiment_dir}" | tail -n 1
}

show_experiment_summary() {
  local exp_dir="$1"
  local total=0
  local with_completions=0
  local done=0
  local pending=0

  mapfile -t checkpoints < <(find_checkpoints "${exp_dir}")
  total=${#checkpoints[@]}
  for ckpt in "${checkpoints[@]}"; do
    if has_completions "${ckpt}"; then
      with_completions=$((with_completions + 1))
      if has_results_for_all_judges "${ckpt}"; then
        done=$((done + 1))
      else
        pending=$((pending + 1))
      fi
    fi
  done

  echo -e "  ${CYAN}${exp_dir#${ROOT_DIR}/}${NC}"
  echo -e "    Checkpoints: ${total} total, ${with_completions} with completions, ${done} fully scored, ${GREEN}${pending} pending${NC}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      ;;
    --list)
      MODE="list"
      shift
      ;;
    --all)
      MODE="all"
      shift
      ;;
    --filter)
      FILTER="$2"
      shift 2
      ;;
    --search-root)
      SEARCH_ROOT="$(normalize_path "$2")"
      SEARCH_ROOT_SET=true
      shift 2
      ;;
    --experiment|-e)
      EXPERIMENT_ARGS+=("$(normalize_path "$2")")
      shift 2
      ;;
    --step)
      STEP="$2"
      shift 2
      ;;
    --last-only)
      LAST_ONLY=true
      shift
      ;;
    --from-file|-f)
      FROM_FILE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --job-subdir)
      JOB_SUBDIR_PREFIX="$2"
      shift 2
      ;;
    --wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --annotators-configs)
      parse_annotators_configs "$2"
      shift 2
      ;;
    -*)
      log_error "Unknown option: $1"
      usage
      ;;
    *)
      CHECKPOINT_ARGS+=("$(normalize_path "$1")")
      shift
      ;;
  esac
done

if [[ ! -d "${SEARCH_ROOT}" ]]; then
  log_error "Search root does not exist: ${SEARCH_ROOT}"
  exit 1
fi

if [[ "${MODE}" == "list" ]]; then
  if [[ ${#ANNOTATORS_CONFIGS[@]} -eq 0 ]]; then
    ANNOTATORS_CONFIGS=("alpaca_eval_gpt4_turbo_fn")
  fi
  log_section "Available experiments (filter: '${FILTER:-<none>}')"
  mapfile -t experiments < <(find_experiments "${FILTER}")
  if [[ ${#experiments[@]} -eq 0 ]]; then
    log_warn "No experiments found"
    exit 0
  fi
  for exp_dir in "${experiments[@]}"; do
    show_experiment_summary "${exp_dir}"
  done
  exit 0
fi

if [[ ${#ANNOTATORS_CONFIGS[@]} -eq 0 ]]; then
  log_error "--annotators-configs is required for scoring"
  exit 1
fi

CHECKPOINTS_TO_SCORE=()

for ckpt in "${CHECKPOINT_ARGS[@]}"; do
  CHECKPOINTS_TO_SCORE+=("${ckpt}")
done

if [[ -n "${FROM_FILE}" ]]; then
  [[ -f "${FROM_FILE}" ]] || {
    log_error "File not found: ${FROM_FILE}"
    exit 1
  }
  while IFS= read -r line; do
    line="$(trim "$(echo "${line}" | sed 's/#.*//')")"
    [[ -z "${line}" ]] && continue
    CHECKPOINTS_TO_SCORE+=("$(normalize_path "${line}")")
  done < "${FROM_FILE}"
fi

for exp_dir in "${EXPERIMENT_ARGS[@]}"; do
  [[ -d "${exp_dir}" ]] || {
    log_error "Experiment directory not found: ${exp_dir}"
    exit 1
  }
  if [[ "${LAST_ONLY}" == true ]]; then
    last_ckpt="$(find_last_checkpoint "${exp_dir}")"
    [[ -n "${last_ckpt}" ]] && CHECKPOINTS_TO_SCORE+=("${last_ckpt}")
  else
    mapfile -t ckpts < <(find_checkpoints "${exp_dir}" "${STEP}")
    CHECKPOINTS_TO_SCORE+=("${ckpts[@]}")
  fi
done

if [[ "${MODE}" == "all" ]]; then
  mapfile -t experiments < <(find_experiments "${FILTER}")
  for exp_dir in "${experiments[@]}"; do
    if [[ "${LAST_ONLY}" == true ]]; then
      last_ckpt="$(find_last_checkpoint "${exp_dir}")"
      [[ -n "${last_ckpt}" ]] && CHECKPOINTS_TO_SCORE+=("${last_ckpt}")
    else
      mapfile -t ckpts < <(find_checkpoints "${exp_dir}" "${STEP}")
      CHECKPOINTS_TO_SCORE+=("${ckpts[@]}")
    fi
  done
fi

if [[ ${#CHECKPOINTS_TO_SCORE[@]} -eq 0 ]]; then
  log_error "No checkpoints specified. Use --help for usage."
  exit 1
fi

if [[ ${#CHECKPOINTS_TO_SCORE[@]} -gt 0 ]]; then
  mapfile -t CHECKPOINTS_TO_SCORE < <(dedupe_preserving_order "${CHECKPOINTS_TO_SCORE[@]}")
fi

VALID_CHECKPOINTS=()
SKIPPED_NO_COMPLETIONS=0
SKIPPED_ALREADY_DONE=0
for ckpt in "${CHECKPOINTS_TO_SCORE[@]}"; do
  [[ -d "${ckpt}" ]] || {
    log_warn "Checkpoint directory not found, skipping: ${ckpt}"
    continue
  }
  if ! has_completions "${ckpt}"; then
    log_warn "No model_outputs.json found, skipping: ${ckpt}"
    SKIPPED_NO_COMPLETIONS=$((SKIPPED_NO_COMPLETIONS + 1))
    continue
  fi
  if has_results_for_all_judges "${ckpt}"; then
    log_info "All requested judges already scored, skipping: ${ckpt}"
    SKIPPED_ALREADY_DONE=$((SKIPPED_ALREADY_DONE + 1))
    continue
  fi
  VALID_CHECKPOINTS+=("${ckpt}")
done

if [[ ${#VALID_CHECKPOINTS[@]} -eq 0 ]]; then
  log_info "Nothing to score. ${SKIPPED_NO_COMPLETIONS} lacked completions, ${SKIPPED_ALREADY_DONE} already done."
  exit 0
fi

log_section "AlpacaEval API Scoring Plan"
echo "Judges:"
for judge in "${ANNOTATORS_CONFIGS[@]}"; do
  echo "  - ${judge} -> $(infer_judge_model "${judge}")"
done
echo "Checkpoints to score: ${#VALID_CHECKPOINTS[@]}"
echo "Skipped (no completions): ${SKIPPED_NO_COMPLETIONS}"
echo "Skipped (already done): ${SKIPPED_ALREADY_DONE}"

if [[ "${DRY_RUN}" == true ]]; then
  log_info "Dry run — not executing. Remove --dry-run to score."
  exit 0
fi

require_runtime_dependencies

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  log_error "OPENAI_API_KEY is not set. Export it before running this script."
  exit 1
fi

echo -e "${YELLOW}This will make OpenAI API calls (costs money). Continue? [y/N]${NC}"
read -r confirm
if [[ "${confirm}" != "y" && "${confirm}" != "Y" ]]; then
  log_info "Aborted."
  exit 0
fi

mkdir -p "${LOG_DIR}"
ANNOTATORS_CSV="$(IFS=,; echo "${ANNOTATORS_CONFIGS[*]}")"

for checkpoint_path in "${VALID_CHECKPOINTS[@]}"; do
  job_id="$(make_job_id "${checkpoint_path}")"
  log_file="${LOG_DIR}/${job_id}.log"
  cmd=(
    python -m qrpo.evals.run_alpaca_gpt
    "checkpoint_path=${checkpoint_path}"
    "annotators_configs=[${ANNOTATORS_CSV}]"
    "job_subdir=${JOB_SUBDIR_PREFIX}/${job_id}"
    "outputs_subdir=shared"
    "resuming.resume=False"
  )
  if [[ -n "${WANDB_PROJECT}" ]]; then
    cmd+=("wandb.project=${WANDB_PROJECT}")
  fi

  echo -e "  ${CYAN}Command:${NC} ${cmd[*]}"
  "${cmd[@]}" 2>&1 | tee "${log_file}"
done
