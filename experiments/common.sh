# Shared helpers for the paper-run registries. Sourced, not executed.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${TUP_DATA_DIR:-$ROOT_DIR/data}"
RUNS="initial dpo rebel rebel-random qrpo qrpo-random bonbon tup-mild tup-mid tup-aggressive"
DRY_RUN=0
EXTRA_ARGS=()  # empty arrays are expanded as ${a[@]+"${a[@]}"}: bash < 4.4 treats "${a[@]}" as unset under set -u

die() { echo "ERROR: $*" >&2; exit 1; }

if [ -n "${NPROC_PER_NODE:-}" ] && [ "$NPROC_PER_NODE" != "4" ]; then
  die "NPROC_PER_NODE must be 4 (paper runs use 4 GPUs x batch 2 x grad-accum 16 = 128); got $NPROC_PER_NODE"
fi
export NPROC_PER_NODE=4
# All inputs are local; offline mode makes the Skywork lookups resolve the pinned cached revision instead of a newer upstream main.
export HF_HUB_OFFLINE=1

parse_common_args() {  # sets DRY_RUN and EXTRA_ARGS from "$@"
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --dry-run) DRY_RUN=1; shift ;;
      --) shift; EXTRA_ARGS=("$@"); break ;;
      *) die "Unknown option: $1" ;;
    esac
  done
}

canonical_dir() {  # <path>: physical path if it is an existing directory, else the path unchanged
  if [ -d "$1" ]; then (cd -P "$1" && pwd); else printf '%s\n' "$1"; fi
}

require_input() {  # <path> <hint command> [-e]   a directory, or with -e any existing path
  test "${3:--d}" "$1" || die "Missing input: $1
Fetch it with: $2"
}

hf_hub_cache() {  # the Hugging Face hub cache dir, resolved as huggingface_hub does
  local hf_home="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
  printf '%s\n' "${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-$hf_home/hub}}"
}

require_cached_model() {  # <pin from scripts/pins.sh>: offline lookups by repo id must resolve the pinned snapshot
  local repo rev repo_dir
  read -r repo rev _ <<< "$1"
  repo_dir="$(hf_hub_cache)/models--${repo//\//--}"
  if [ "$(cat "$repo_dir/refs/main" 2> /dev/null)" != "$rev" ] || [ ! -d "$repo_dir/snapshots/$rev" ]; then
    die "Missing input: $repo at revision $rev in the Hugging Face cache ($repo_dir, with refs/main naming that snapshot)
Fetch it with: bash scripts/download_models.sh"
  fi
}

require_eval_inputs() {  # <dataset_id> <default ref model key> <launcher args...>: what the eval stages read after training
  local dataset_id="$1" ref_key="$2" a prev="" family suffix subset rm
  local skip_split=0 skip_alpaca=0 skip_rescore=0
  shift 2
  for a in "$@"; do
    case "$a" in
      --skip-eval-split) skip_split=1 ;;
      --skip-alpaca-eval) skip_alpaca=1 ;;
      --skip-rescore) skip_rescore=1 ;;
    esac
    if [ "$prev" = "--ref-model-key" ]; then ref_key="$a"; fi
    prev="$a"
  done
  local shared="$DATA_DIR/shared"
  local pools="$shared/datasets/alpaca-ref-completions/$ref_key"
  local score_hint="bash scripts/score_reference_pools.sh (or: bash scripts/score_reference_pools.sh --use-hosted)"
  if [ "$skip_split" != "1" ] || [ "$skip_alpaca" != "1" ]; then
    require_input "$shared/models/armorm" "bash scripts/download_models.sh"
  fi
  if [ "$skip_split" != "1" ]; then
    # Same derivation as run_online_eval.py's eval split keys path: <family>-armorm from the dataset id.
    IFS=- read -r _ _ family suffix _ <<< "$dataset_id"
    for subset in checkpoint-selection reward-reporting; do
      require_input "$shared/datasets-with-chosen-rewards/merged/$family-$suffix/eval-$subset-indices.npy" \
        "bash scripts/download_data.sh" -e
    done
  fi
  if [ "$skip_alpaca" != "1" ]; then
    require_input "$shared/datasets/alpaca" "bash scripts/download_data.sh"
    require_input "$pools/raw-completions" "bash scripts/download_data.sh"
    require_input "$pools/armorm" "$score_hint"
    if [ "$skip_rescore" != "1" ]; then
      for rm in skywork_v2_llama skywork_v2_qwen; do
        require_input "$pools/$rm" "$score_hint"
      done
    fi
  fi
  if [ "$skip_rescore" != "1" ]; then
    source "$ROOT_DIR/scripts/pins.sh"
    require_cached_model "$PIN_SKYWORK_LLAMA"
    require_cached_model "$PIN_SKYWORK_QWEN"
  fi
}

launch() {  # <label> <launcher> <dataset_id> <model_dir_name> <launcher flags...>
  local label="$1" launcher="$2" dataset_id="$3" model_dir="$4"
  shift 4
  local dataset_path="$DATA_DIR/shared/datasets-with-ref-rewards/merged/$dataset_id"
  local model_path="$DATA_DIR/shared/models/$model_dir"
  local -a ds_flag=()
  # Only the Llama launcher accepts --dataset-path; the Mistral launcher always reads
  # $ROOT_DIR/data/shared/datasets-with-ref-rewards/merged/$dataset_id.
  if [ "$launcher" = "$ROOT_DIR/run_llama_experiment.sh" ]; then
    ds_flag=(--dataset-path "$dataset_path")
  fi
  if [ "$DRY_RUN" != "1" ]; then
    require_input "$model_path" "bash scripts/download_models.sh"
    require_input "$dataset_path" "bash scripts/download_data.sh"
    local default_ref_key=mistral  # the launchers' own --ref-model-key defaults
    if [ "$launcher" = "$ROOT_DIR/run_llama_experiment.sh" ]; then
      default_ref_key=llama
    fi
    require_eval_inputs "$dataset_id" "$default_ref_key" "$@" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    if [ "$(canonical_dir "$DATA_DIR")" != "$(canonical_dir "$ROOT_DIR/data")" ]; then
      # The eval stages of both launchers read config.data_dir = $ROOT_DIR/data
      # (alpaca prompts, split indices, AlpacaEval reference scores).
      die "TUP_DATA_DIR is only supported by the download scripts and dry runs, because the eval stages read $ROOT_DIR/data. Link it instead, then unset TUP_DATA_DIR: ln -s \"$(canonical_dir "$DATA_DIR")\" \"$ROOT_DIR/data\" && unset TUP_DATA_DIR"
    fi
  fi
  echo "== $label =="
  if ! DRY_RUN="$DRY_RUN" bash "${LAUNCHER_OVERRIDE:-$launcher}" \
      --dataset-id "$dataset_id" ${ds_flag[@]+"${ds_flag[@]}"} --model-path "$model_path" \
      "$@" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; then
    echo "FAILED: $label" >&2
    return 1
  fi
}

run_selected() {  # <entry function> <setting> <run|all>
  local fn="$1" setting="$2" run="$3" r
  if [ "$run" = "all" ]; then
    for r in $RUNS; do "$fn" "$setting" "$r" || exit 1; done
  else
    case " $RUNS " in *" $run "*) ;; *) die "Unknown run '$run'. Valid runs: $RUNS all" ;; esac
    "$fn" "$setting" "$run" || exit 1
  fi
}
