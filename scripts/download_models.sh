#!/usr/bin/env bash
# Download the models used by the paper runs at pinned revisions.
# Usage: bash scripts/download_models.sh [--llama] [--mistral] [--sft-base] [--dry-run]
#   --llama / --mistral: that policy (Tulu 3 SFT / our Mistral SFT) plus the three reward models (ArmoRM, Skywork-V2 Llama/Qwen)
#   --sft-base: Mistral-7B-Instruct-v0.2, the start model for scripts/train_mistral_sft.sh (alone, it fetches nothing else)
#   no selection flags = --llama --mistral
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${TUP_DATA_DIR:-$ROOT_DIR/data}"
source "$ROOT_DIR/scripts/pins.sh"
DRY_RUN=0; WANT_LLAMA=0; WANT_MISTRAL=0; WANT_SFT_BASE=0
for a in "$@"; do
  case "$a" in
    --llama) WANT_LLAMA=1 ;; --mistral) WANT_MISTRAL=1 ;; --sft-base) WANT_SFT_BASE=1 ;;
    --dry-run) DRY_RUN=1 ;; *) echo "ERROR: unknown option $a" >&2; exit 1 ;;
  esac
done
[ "$WANT_LLAMA$WANT_MISTRAL$WANT_SFT_BASE" = "000" ] && { WANT_LLAMA=1; WANT_MISTRAL=1; }

fetch() {  # "<repo> <rev> <type> <dest>" [extra hf args...]
  local repo rev type dest; read -r repo rev type dest <<< "$1"; shift
  if [ "$rev" = "UNPUBLISHED" ]; then
    echo "ERROR: $repo is not published yet; see README 'Artifacts'." >&2; [ "$DRY_RUN" = "1" ] || exit 1
  fi
  local -a cmd=(hf download "$repo" --revision "$rev" --repo-type "$type")
  [ "$dest" = "HF_CACHE" ] || cmd+=(--local-dir "$DATA_DIR/$dest")
  cmd+=("$@")
  printf '+'; printf ' %q' "${cmd[@]}"; printf '\n'
  [ "$DRY_RUN" = "1" ] || "${cmd[@]}"
  [ "$dest" = "HF_CACHE" ] && pin_main_ref "$repo" "$rev"
  return 0
}

# The launchers look cached reward models up by repo id without a revision, which needs refs/main.
# `hf download <repo> --revision <sha>` writes no ref, so point main at the pinned snapshot.
pin_main_ref() {  # <repo> <sha>
  local repo="$1" sha="$2"
  if [ "$DRY_RUN" = "1" ]; then
    echo "+ pin refs/main $repo $sha"
    return 0
  fi
  local hub refs
  hub="$(python -c 'from huggingface_hub.constants import HF_HUB_CACHE; print(HF_HUB_CACHE)')" \
    || { echo "ERROR: cannot resolve the Hugging Face hub cache dir (is huggingface_hub installed in this python?)" >&2; exit 1; }
  refs="$hub/models--${repo//\//--}/refs"
  mkdir -p "$refs"
  printf '%s' "$sha" > "$refs/main"
}

if [ "$WANT_LLAMA" = "1" ] || [ "$WANT_MISTRAL" = "1" ]; then  # reward models for evaluation
  fetch "$PIN_ARMORM"
  fetch "$PIN_SKYWORK_LLAMA"
  fetch "$PIN_SKYWORK_QWEN"
fi
[ "$WANT_LLAMA" = "1" ] && fetch "$PIN_TULU"
[ "$WANT_MISTRAL" = "1" ] && fetch "$PIN_MISTRAL_SFT"
[ "$WANT_SFT_BASE" = "1" ] && fetch "$PIN_MISTRAL_BASE" --include "*.json" "*.safetensors" "tokenizer*"
exit 0
