#!/usr/bin/env bash
# Judge one checkpoint's seed-0 AlpacaEval generations with gpt-4o (paid OpenAI API; asks before spending).
# Usage: bash scripts/judge_alpaca_gpt4o.sh <checkpoint_dir> [--dry-run]
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${1:?Usage: bash scripts/judge_alpaca_gpt4o.sh <checkpoint_dir> [--dry-run]}"
DRY_RUN=0; [ "${2:-}" = "--dry-run" ] && DRY_RUN=1
OUTPUTS="$CKPT/online_evals_0/alpaca_eval/model_outputs.json"
[ -f "$OUTPUTS" ] || { echo "ERROR: missing $OUTPUTS (online_evals_0/alpaca_eval/model_outputs.json); run the experiment's eval stage first." >&2; exit 1; }
CKPT_LIST="$(mktemp)"; trap 'rm -f "$CKPT_LIST"' EXIT
CKPT_REAL="$(realpath "$CKPT")"
echo "$CKPT_REAL" > "$CKPT_LIST"
CMD=(env WANDB_MODE=disabled bash "$ROOT_DIR/run_alpaca_eval_api.sh" --annotators-configs alpaca_eval_gpt4o_fn --from-file "$CKPT_LIST" --search-root "$CKPT_REAL")
printf '+'; printf ' %q' "${CMD[@]}"; printf '\n'
[ "$DRY_RUN" = "1" ] && exit 0
[ -n "${OPENAI_API_KEY:-}" ] || { echo "ERROR: export OPENAI_API_KEY first." >&2; exit 1; }
"${CMD[@]}"
