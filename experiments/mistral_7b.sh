#!/usr/bin/env bash
# Mistral 7B SFT runs on Magpie Air reported in the paper (Tables 3, 6, 11).
# Usage: bash experiments/mistral_7b.sh <run|all> [--dry-run] [-- <extra launcher args>]
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
LAUNCHER="$ROOT_DIR/run_mistral_experiment.sh"
RANDOM_DS=mistral-sft-magpieair-armorm-temp1-ref50-offpolicy2random-armorm
BEST_DS=mistral-sft-magpieair-armorm-temp1-ref50-offpolicy2best-armorm
COMMON=(--ref-model-key mistral-sft --scheduler-max-steps 764)
TUP=(--loss-type ours --beta 0.01 --num-ref-rewards 6 --qrpo-sample-selector random --use-discrete-partition False --learning-rate 1e-6)

entry() {  # <setting=mistral> <run>   values: spec §7.2
  local l="mistral/$2"
  case "$2" in
    initial)        launch "$l" "$LAUNCHER" "$RANDOM_DS" mistral-sft "${COMMON[@]}" "${TUP[@]}" --binomial-lambda 0.5 --max-steps 764 --eval-base-only ;;
    dpo)            launch "$l" "$LAUNCHER" "$RANDOM_DS" mistral-sft "${COMMON[@]}" --loss-type dpo   --learning-rate 3e-7 --beta 0.03 --num-ref-rewards 6 --qrpo-sample-selector both --max-steps 480 ;;
    rebel)          launch "$l" "$LAUNCHER" "$BEST_DS"   mistral-sft "${COMMON[@]}" --loss-type rebel --learning-rate 1e-7 --beta 1e-4 --num-ref-rewards 6 --qrpo-sample-selector both --max-steps 480 ;;
    rebel-random)   launch "$l" "$LAUNCHER" "$RANDOM_DS" mistral-sft "${COMMON[@]}" --loss-type rebel --learning-rate 3e-7 --beta 1e-4 --num-ref-rewards 6 --qrpo-sample-selector both --max-steps 320 ;;
    qrpo)           launch "$l" "$LAUNCHER" "$BEST_DS"   mistral-sft "${COMMON[@]}" --loss-type qrpo  --learning-rate 1e-7 --beta 3e-3 --num-ref-rewards 1 --qrpo-sample-selector both --max-steps 764 ;;
    qrpo-random)    launch "$l" "$LAUNCHER" "$RANDOM_DS" mistral-sft "${COMMON[@]}" --loss-type qrpo  --learning-rate 1e-6 --beta 3e-3 --num-ref-rewards 1 --qrpo-sample-selector both --max-steps 764 ;;
    bonbon)         launch "$l" "$LAUNCHER" "$BEST_DS"   mistral-sft "${COMMON[@]}" --loss-type bonbon --learning-rate 1e-7 --beta 1e-3 --num-ref-rewards 6 --binomial-lambda 0.5 --qrpo-sample-selector both --use-discrete-partition False --max-steps 480 ;;
    tup-mild)       launch "$l" "$LAUNCHER" "$RANDOM_DS" mistral-sft "${COMMON[@]}" "${TUP[@]}" --binomial-lambda 0.2 --max-steps 764 ;;
    tup-mid)        launch "$l" "$LAUNCHER" "$RANDOM_DS" mistral-sft "${COMMON[@]}" "${TUP[@]}" --binomial-lambda 0.5 --max-steps 764 ;;
    tup-aggressive) launch "$l" "$LAUNCHER" "$RANDOM_DS" mistral-sft "${COMMON[@]}" "${TUP[@]}" --binomial-lambda 0.8 --max-steps 764 ;;
    *) die "Unknown run '$2'" ;;
  esac
}

[ "$#" -ge 1 ] || die "Usage: bash experiments/mistral_7b.sh <run|all> [--dry-run] [-- <extra launcher args>]"
RUN="$1"; shift
parse_common_args "$@"
run_selected entry mistral "$RUN"
