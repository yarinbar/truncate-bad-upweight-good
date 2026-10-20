#!/usr/bin/env bash
# Llama 8B Tulu 3 SFT runs reported in the paper (Tables 1, 2, 4, 5, 9, 10).
# Usage: bash experiments/llama_8b.sh <uf|magpie|all> <run|all> [--dry-run] [-- <extra launcher args>]
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
LAUNCHER="$ROOT_DIR/run_llama_experiment.sh"
UF_RANDOM=llama-nosft-ultrafeedback-armorm-temp1-ref50-offpolicy2random-armorm
UF_BEST=llama-nosft-ultrafeedback-armorm-temp1-ref50-offpolicy2best-armorm
MA_RANDOM=llama-nosft-magpieair-armorm-temp1-ref50-offpolicy2random-armorm
MA_BEST=llama-nosft-magpieair-armorm-temp1-ref50-offpolicy2best-armorm
TUP=(--loss-type ours --beta 0.01 --num-ref-rewards 6 --qrpo-sample-selector random --use-discrete-partition False)
BONBON=(--loss-type bonbon --beta 1e-3 --num-ref-rewards 6 --binomial-lambda 0.5 --qrpo-sample-selector both --use-discrete-partition False)

entry() {  # <setting> <run>   values: spec §7.2 (verified against archived configs)
  local l="$1/$2"
  case "$l" in
    uf/initial)            launch "$l" "$LAUNCHER" "$UF_RANDOM" llama "${TUP[@]}" --learning-rate 1e-7 --binomial-lambda 0.5 --scheduler-max-steps 476 --max-steps 400 --eval-base-only ;;
    uf/dpo)                launch "$l" "$LAUNCHER" "$UF_BEST"   llama --loss-type dpo   --learning-rate 3e-7 --beta 0.01 --num-ref-rewards 6 --qrpo-sample-selector both   --scheduler-max-steps 476 --max-steps 300 ;;
    uf/rebel)              launch "$l" "$LAUNCHER" "$UF_BEST"   llama --loss-type rebel --learning-rate 3e-7 --beta 1e-6 --num-ref-rewards 6 --qrpo-sample-selector both   --scheduler-max-steps 476 --max-steps 200 ;;
    uf/rebel-random)       launch "$l" "$LAUNCHER" "$UF_RANDOM" llama --loss-type rebel --learning-rate 1e-6 --beta 1e-6 --num-ref-rewards 6 --qrpo-sample-selector random --scheduler-max-steps 476 --max-steps 100 ;;
    uf/qrpo)               launch "$l" "$LAUNCHER" "$UF_BEST"   llama --loss-type qrpo  --learning-rate 3e-7 --beta 3e-4 --num-ref-rewards 3 --qrpo-sample-selector both   --scheduler-max-steps 476 --max-steps 100 ;;
    uf/qrpo-random)        launch "$l" "$LAUNCHER" "$UF_RANDOM" llama --loss-type qrpo  --learning-rate 3e-7 --beta 3e-4 --num-ref-rewards 3 --qrpo-sample-selector random --scheduler-max-steps 476 --max-steps 100 ;;
    uf/bonbon)             launch "$l" "$LAUNCHER" "$UF_BEST"   llama "${BONBON[@]}" --learning-rate 1e-6 --scheduler-max-steps 476 --max-steps 476 ;;
    uf/tup-mild)           launch "$l" "$LAUNCHER" "$UF_RANDOM" llama "${TUP[@]}" --learning-rate 1e-7 --binomial-lambda 0.2 --scheduler-max-steps 476 --max-steps 400 ;;
    uf/tup-mid)            launch "$l" "$LAUNCHER" "$UF_RANDOM" llama "${TUP[@]}" --learning-rate 1e-7 --binomial-lambda 0.5 --scheduler-max-steps 476 --max-steps 400 ;;
    uf/tup-aggressive)     launch "$l" "$LAUNCHER" "$UF_RANDOM" llama "${TUP[@]}" --learning-rate 1e-7 --binomial-lambda 0.8 --scheduler-max-steps 476 --max-steps 476 ;;
    magpie/initial)        launch "$l" "$LAUNCHER" "$MA_RANDOM" llama "${TUP[@]}" --learning-rate 1e-7 --binomial-lambda 0.5 --scheduler-max-steps 764 --max-steps 320 --eval-base-only ;;
    magpie/dpo)            launch "$l" "$LAUNCHER" "$MA_RANDOM" llama --loss-type dpo   --learning-rate 1e-6 --beta 0.01 --num-ref-rewards 6 --qrpo-sample-selector both --scheduler-max-steps 480 --max-steps 480 ;;
    magpie/rebel)          launch "$l" "$LAUNCHER" "$MA_BEST"   llama --loss-type rebel --learning-rate 1e-7 --beta 1e-6 --num-ref-rewards 6 --qrpo-sample-selector both --scheduler-max-steps 640 --max-steps 640 ;;
    magpie/rebel-random)   launch "$l" "$LAUNCHER" "$MA_RANDOM" llama --loss-type rebel --learning-rate 1e-6 --beta 1e-4 --num-ref-rewards 6 --qrpo-sample-selector both --scheduler-max-steps 320 --max-steps 320 ;;
    magpie/qrpo)           launch "$l" "$LAUNCHER" "$MA_BEST"   llama --loss-type qrpo  --learning-rate 1e-7 --beta 1e-3 --num-ref-rewards 1 --qrpo-sample-selector both --scheduler-max-steps 320 --max-steps 320 ;;
    magpie/qrpo-random)    launch "$l" "$LAUNCHER" "$MA_RANDOM" llama --loss-type qrpo  --learning-rate 1e-7 --beta 1e-3 --num-ref-rewards 1 --qrpo-sample-selector both --scheduler-max-steps 320 --max-steps 320 ;;
    magpie/bonbon)         launch "$l" "$LAUNCHER" "$MA_BEST"   llama "${BONBON[@]}" --learning-rate 1e-6 --scheduler-max-steps 764 --max-steps 764 ;;
    magpie/tup-mild)       launch "$l" "$LAUNCHER" "$MA_RANDOM" llama "${TUP[@]}" --learning-rate 1e-7 --binomial-lambda 0.2 --scheduler-max-steps 764 --max-steps 320 ;;
    magpie/tup-mid)        launch "$l" "$LAUNCHER" "$MA_RANDOM" llama "${TUP[@]}" --learning-rate 1e-7 --binomial-lambda 0.5 --scheduler-max-steps 764 --max-steps 320 ;;
    magpie/tup-aggressive) launch "$l" "$LAUNCHER" "$MA_RANDOM" llama "${TUP[@]}" --learning-rate 1e-7 --binomial-lambda 0.8 --scheduler-max-steps 764 --max-steps 320 ;;
    *) die "Unknown setting/run '$l'" ;;
  esac
}

[ "$#" -ge 2 ] || die "Usage: bash experiments/llama_8b.sh <uf|magpie|all> <run|all> [--dry-run] [-- <extra launcher args>]"
SETTING="$1"; RUN="$2"; shift 2
parse_common_args "$@"
case "$SETTING" in
  uf|magpie) run_selected entry "$SETTING" "$RUN" ;;
  all) run_selected entry uf "$RUN"; run_selected entry magpie "$RUN" ;;
  *) die "Unknown setting '$SETTING'. Valid settings: uf magpie all" ;;
esac
