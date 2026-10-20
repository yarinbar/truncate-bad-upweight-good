# Truncate Bad, Upweight Good (TUP)

Code to reproduce **"Truncate Bad, Upweight Good: BoN-Style Distillation via Rank-Based Classification"**
(Yarin Bar, Yaniv Romano; [arXiv:2608.19748](https://arxiv.org/abs/2608.19748)).

TUP trains a policy fully offline with a binary cross-entropy loss whose soft labels are shifted-truncated
in-pool win-rates and whose logits are policy-to-reference log-ratios plus a closed-form intercept
β·log Z(λ, β). In the code TUP is `loss_type=ours` (in `src/qrpo/trainers/qrpo.py`), inside the
[QRPO](https://github.com/CLAIRE-Labo/quantile-reward-policy-optimization) training framework this work builds on.

## What this repository reproduces

| Paper | Setting | Command |
|---|---|---|
| Tables 1, 2, 4, 9 | Llama 8B Tülu 3 SFT on UltraFeedback | `bash experiments/llama_8b.sh uf all` |
| Tables 1, 2, 5, 10 | Llama 8B Tülu 3 SFT on Magpie Air | `bash experiments/llama_8b.sh magpie all` |
| Tables 3, 6, 11 | Mistral 7B SFT on Magpie Air | `bash experiments/mistral_7b.sh all` |
| Figs 2, 3, 5, 6, 7; App. B.3 | Static (training-free) experiments | `notebooks/static_experiments.ipynb` |

Each setting has the runs `initial dpo rebel rebel-random qrpo qrpo-random bonbon tup-mild tup-mid tup-aggressive`,
e.g. `bash experiments/llama_8b.sh uf tup-mid`. Add `--dry-run` to print the exact training command.
Each run trains to the checkpoint reported in Tables 9–11 and evaluates it.

## Hardware

One node with 4 GPUs (the paper used 4× H200 141 GB and 64 CPU cores). The registries require exactly 4
processes because the effective batch size (2 per device × 16 accumulation × 4 GPUs = 128) is part of the setup.
Disk: about 292 GB of downloads for all runs (about 185 GB for TUP only), plus tens of GB of `datasets` cache per dataset.

## Setup

```bash
bash install.sh                        # conda env "tup" with the paper's package versions (requirements.txt)
conda activate tup
bash scripts/download_models.sh        # Tülu 3 SFT, our Mistral SFT, ArmoRM, Skywork-V2 Llama/Qwen (pinned revisions)
bash scripts/download_data.sh          # QRPO reference sets, AlpacaEval, our AlpacaEval reference completions
bash scripts/score_reference_pools.sh  # Skywork (and ArmoRM) scores of the reference pools, needed for LC metrics
```

`requirements.txt` pins the paper environment exactly, except `tinycss2` (1.4.0 instead of 1.5.1, to resolve a `bleach[css]` conflict; only the notebook tooling uses it).
`download_data.sh --tup-only` fetches only the `offpolicy2random` sets that TUP uses.
`score_reference_pools.sh` scores about 600K eval-split reference completions (all six training sets) with both Skywork models and the AlpacaEval reference pools with ArmoRM and both Skywork models, about 1.3M reward-model passes in total; this takes several hours on 4 GPUs.
`score_reference_pools.sh --use-hosted` links our exact score files instead of recomputing them (fast).
The download scripts honour `TUP_DATA_DIR` (default `data/`). Training and evaluation read from `data/` only, so if you downloaded elsewhere, link it and unset the variable: `ln -s "$(realpath "$TUP_DATA_DIR")" data && unset TUP_DATA_DIR`.
The experiment registries run with `HF_HUB_OFFLINE=1`: every input is local, and `download_models.sh` points the Hugging Face cache's `main` ref of the two Skywork models at their pinned revisions.
If you call `run_llama_experiment.sh`, `run_mistral_experiment.sh` or `run_rescore_multi_rm.sh` directly instead of through `experiments/*.sh`, run `export HF_HUB_OFFLINE=1` first so that the pinned Skywork revisions are used.
When you call `run_rescore_multi_rm.sh` directly on a trained run, also set `CHECKPOINT_STEPS` to the run's stop step (for example `CHECKPOINT_STEPS=400`): the eval stage links the base model into the run as `checkpoint-0`, and without a step filter the rescoring picks that link up.

## Running experiments

```bash
bash experiments/llama_8b.sh uf tup-mid        # one run
bash experiments/llama_8b.sh magpie all        # every reported run of a setting
bash experiments/mistral_7b.sh all
bash scripts/judge_alpaca_gpt4o.sh outputs/shared/train_qrpo/chat-baseline/<run>/checkpoints/<hash>/checkpoint-<N>   # optional: paid OpenAI API, needs OPENAI_API_KEY
```

W&B logging is off unless you set `WANDB_MODE=online`.

## Static notebook

`notebooks/static_experiments.ipynb` runs on CPU from the committed `notebooks/static_data.npz`; open it from `notebooks/` with the `tup` environment's kernel.
To rebuild the `.npz` from your own downloads, run `python scripts/export_static_data.py` after `download_data.sh` and `score_reference_pools.sh`. It needs only the Llama UltraFeedback and Magpie Air `offpolicy2random` sets with their Skywork eval-split scores, so `download_data.sh --tup-only` is enough.
Built from scores you computed locally, the `.npz` may differ slightly from the committed one (rescoring noise); built after `score_reference_pools.sh --use-hosted`, it reproduces it.
The App. B.3 cell is fixed to its reported configuration (Llama UltraFeedback, ArmoRM proxy, Skywork-Llama oracle).

## Where the numbers are

`<run>` is the run directory that `--dry-run` prints as `RUN_ROOT=` (an auto-generated name built from the run's settings), `<hash>` is the single config-hash directory inside its `checkpoints/`, and `<dataset_id>` is the training-set id used by the registries (e.g. `llama-nosft-ultrafeedback-armorm-temp1-ref50-offpolicy2random-armorm`; see `experiments/*.sh`).

Under `outputs/shared/train_qrpo/chat-baseline/<run>/checkpoints/<hash>/checkpoint-<N>/`:

| Table column | File | Key |
|---|---|---|
| ArmoRM LC reward (in-dataset / AlpacaEval) | `online_evals_logs_eval_split.json` / `online_evals_logs_alpaca_eval.json` | `online_eval/<eval_type>/reporting/lc_lcmodel_avg_reward_mean` (± `_std`) |
| Raw reward | same files | `…/reporting/avg_reward_mean` |
| Length (tokens) | same files | `…/reporting/avg_reward_tokens_len_mean` |
| Skywork-Llama / Skywork-Qwen | `skywork_v2_llama_online_evals_logs_*.json` / `skywork_v2_qwen_…` | same keys |
| GPT-4o LC win / win / length (characters) | `alpaca_eval_gpt4o_fn/leaderboard.csv` | `length_controlled_winrate` ± `lc_standard_error`, `win_rate` ± `standard_error`, `avg_length` |

"Initial" rows are the same files in `outputs/shared/train_qrpo/chat-baseline/initial-<dataset_id>/checkpoints/base/checkpoint-0/` (the base model); `initial` runs produce only those.
A trained run evaluates and rescores only its reported checkpoint, `checkpoint-<N>` (the stop checkpoint); it does not evaluate `checkpoint-0`.
Reward-model ± is the sample standard deviation over 3 generation seeds; GPT-4o ± is alpaca_eval's standard error.

## Optional: retrain the Mistral SFT model

```bash
bash scripts/download_models.sh --sft-base
bash scripts/download_data.sh --setting mistral-magpie --sft
bash scripts/train_mistral_sft.sh      # 1 epoch, lr 5e-7, batch 128, seed 42 -> data/shared/models/mistral-sft
```

## Artifacts

Models and data are downloaded from the Hugging Face Hub at the revisions pinned in `scripts/pins.sh`.
Our own artifacts: [`yarinbar/tup-mistral-7b-sft-magpieair`](https://huggingface.co/yarinbar/tup-mistral-7b-sft-magpieair)
and [`yarinbar/tup-eval-artifacts`](https://huggingface.co/datasets/yarinbar/tup-eval-artifacts).

## Citation

```bibtex
@article{bar2026tup,
  title   = {Truncate Bad, Upweight Good: BoN-Style Distillation via Rank-Based Classification},
  author  = {Bar, Yarin and Romano, Yaniv},
  journal = {arXiv preprint arXiv:2608.19748},
  year    = {2026}
}
```

## License

MIT (see `LICENSE` and `NOTICE`); this work builds on QRPO (MIT) and TRL (Apache-2.0, see `LICENSES/Apache-2.0.txt`).
