#!/usr/bin/env bash
# Download the datasets used by the paper runs at pinned revisions, then write the eval split indices.
# Usage: bash scripts/download_data.sh [--setting llama-uf|llama-magpie|mistral-magpie|all] [--tup-only] [--sft] [--dry-run]
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${TUP_DATA_DIR:-$ROOT_DIR/data}"
source "$ROOT_DIR/scripts/pins.sh"
SETTING=all; TUP_ONLY=0; SFT=0; DRY_RUN=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --setting) SETTING="$2"; shift 2 ;; --tup-only) TUP_ONLY=1; shift ;;
    --sft) SFT=1; shift ;; --dry-run) DRY_RUN=1; shift ;;
    *) echo "ERROR: unknown option $1" >&2; exit 1 ;;
  esac
done

fetch() {  # "<repo> <rev> <type> <dest>"
  local repo rev type dest; read -r repo rev type dest <<< "$1"
  if [ "$rev" = "UNPUBLISHED" ]; then
    echo "ERROR: $repo is not published yet; see README 'Artifacts'." >&2; [ "$DRY_RUN" = "1" ] || exit 1
  fi
  local -a cmd=(hf download "$repo" --revision "$rev" --repo-type "$type" --local-dir "$DATA_DIR/$dest")
  printf '+'; printf ' %q' "${cmd[@]}"; printf '\n'
  [ "$DRY_RUN" = "1" ] || "${cmd[@]}"
}
pair() { fetch "$1"; [ "$TUP_ONLY" = "1" ] || fetch "$2"; }

case "$SETTING" in
  llama-uf) pair "$PIN_LLAMA_UF_RANDOM" "$PIN_LLAMA_UF_BEST" ;;
  llama-magpie) pair "$PIN_LLAMA_MA_RANDOM" "$PIN_LLAMA_MA_BEST" ;;
  mistral-magpie) pair "$PIN_MISTRAL_MA_RANDOM" "$PIN_MISTRAL_MA_BEST" ;;
  all) pair "$PIN_LLAMA_UF_RANDOM" "$PIN_LLAMA_UF_BEST"; pair "$PIN_LLAMA_MA_RANDOM" "$PIN_LLAMA_MA_BEST"
       pair "$PIN_MISTRAL_MA_RANDOM" "$PIN_MISTRAL_MA_BEST" ;;
  *) echo "ERROR: unknown --setting $SETTING (llama-uf|llama-magpie|mistral-magpie|all)" >&2; exit 1 ;;
esac
[ "$SFT" = "1" ] && fetch "$PIN_SFT_DATA"

# AlpacaEval prompts: saved with save_to_disk exactly as run_online_eval.py expects (split "eval", 805 rows).
read -r ALPACA_REPO ALPACA_REV _ _ <<< "$PIN_ALPACA"
printf '+ hf download %s --revision %s (alpaca_eval eval split -> %s)\n' "$ALPACA_REPO" "$ALPACA_REV" "$DATA_DIR/shared/datasets/alpaca"
if [ "$DRY_RUN" != "1" ]; then
  python - "$ALPACA_REPO" "$ALPACA_REV" "$DATA_DIR/shared/datasets/alpaca" <<'PY'
import sys, datasets
repo, rev, dest = sys.argv[1:]
ds = datasets.load_dataset(repo, "alpaca_eval", revision=rev, trust_remote_code=True)
assert len(ds["eval"]) == 805, len(ds["eval"])
ds.save_to_disk(dest)
PY
fi

# Hosted eval artifacts: AlpacaEval reference completions (+ scores) and Skywork eval-split reference scores.
fetch "$PIN_EVAL_ARTIFACTS"
if [ "$DRY_RUN" != "1" ]; then
  mkdir -p "$DATA_DIR/shared/datasets/alpaca-ref-completions"
  for m in llama mistral-sft; do
    mkdir -p "$DATA_DIR/shared/datasets/alpaca-ref-completions/$m"
    ln -sfn "$DATA_DIR/shared/tup-eval-artifacts/alpaca-ref-completions/$m/raw-completions" \
      "$DATA_DIR/shared/datasets/alpaca-ref-completions/$m/raw-completions"
  done
  python "$ROOT_DIR/scripts/make_eval_split_indices.py" --data-dir "$DATA_DIR"
fi
