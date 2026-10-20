#!/usr/bin/env python3
"""
Re-score existing evaluation completions with a different reward model.

Phase 1: Score completions with the new RM and prepare the prefixed directory
         structure so the existing eval pipeline can pick it up.
Phase 2: Invoke the existing run_online_eval.py which skips generation,
         reward scoring, and KL computation once those artifacts exist.
"""

import argparse
import logging
import multiprocessing
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import datasets
import torch
import yaml
from tqdm import tqdm

from qrpo.evals.checkpoint_filter import (
    normalize_checkpoint_steps,
    select_checkpoint_paths,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REWARD_COLUMNS = ["reward", "reward_tokens", "reward_tokens_len", "reward_text"]


def _extract_scores(out, attention_mask, is_causal_rm):
    """Extract reward scores from model output."""
    if is_causal_rm:
        last_pos = attention_mask.sum(dim=1) - 1
        return out.logits[torch.arange(out.logits.size(0)), last_pos, 0]
    scores = out.score if hasattr(out, "score") else out.logits
    return scores.squeeze(-1)


def _is_causal_lm_reward_model(model_path):
    """Check if a reward model uses CausalLM architecture."""
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        architectures = getattr(config, "architectures", [])
        return any("CausalLM" in arch for arch in architectures)
    except Exception:
        return False


def _get_single_gpu_mem_gb():
    """Get single GPU memory in GB via nvidia-smi."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(out.stdout.strip().split("\n")[0]) / 1024
    except Exception:
        return 80.0


def _model_needs_sharding(model_path):
    """Check if a model is too large for a single GPU in bf16."""
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        h = getattr(config, "hidden_size", 0)
        n = getattr(config, "num_hidden_layers", 0)
        v = getattr(config, "vocab_size", 0)
        inter = getattr(config, "intermediate_size", h * 4)
        est_params = n * (4 * h * h + 3 * h * inter) + v * h * 2
        est_gb = est_params * 2 / (1024**3)
        gpu_mem_gb = _get_single_gpu_mem_gb()
        logger.info(
            f"Model size estimate: {est_gb:.1f} GB (bf16), "
            f"single GPU memory: {gpu_mem_gb:.1f} GB"
        )
        return est_gb > gpu_mem_gb * 0.85
    except Exception as exc:
        logger.warning(f"Could not estimate model size: {exc}")
        return False


def compute_rewards_worker(
    chunk_data,
    reward_model_path,
    batch_size,
    max_seq_length,
    gpu_id,
    is_causal_rm=False,
):
    """Score a chunk of chat-formatted completions on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    model_cls = (
        AutoModelForCausalLM
        if is_causal_rm
        else AutoModelForSequenceClassification
    )
    model = model_cls.from_pretrained(
        reward_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(reward_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_rewards, all_tokens, all_tokens_len, all_texts = [], [], [], []
    for i in tqdm(range(0, len(chunk_data), batch_size), desc=f"GPU {gpu_id}"):
        batch = chunk_data[i : i + batch_size]
        tok = tokenizer.apply_chat_template(
            batch,
            return_tensors="pt",
            padding=True,
            return_dict=True,
            truncation=True,
            max_length=max_seq_length,
        )
        for j in range(len(tok["attention_mask"])):
            mask = tok["attention_mask"][j]
            tlen = int(torch.sum(mask).item())
            toks = tok["input_ids"][j][:tlen].tolist()
            all_tokens.append(toks)
            all_tokens_len.append(tlen)
            all_texts.append(tokenizer.decode(toks, skip_special_tokens=False))
        with torch.no_grad():
            attn = tok["attention_mask"].to("cuda:0")
            out = model(
                input_ids=tok["input_ids"].to("cuda:0"),
                attention_mask=attn,
            )
        scores = _extract_scores(out, attn, is_causal_rm)
        all_rewards.extend(scores.cpu().tolist())
    return all_rewards, all_tokens, all_tokens_len, all_texts


def score_parallel(
    completions_list,
    reward_model_path,
    batch_size,
    max_seq_length,
    num_gpus,
    is_causal_rm=False,
):
    """Score completions across multiple GPUs via ProcessPoolExecutor."""
    n = min(num_gpus, len(completions_list))
    chunk_size = len(completions_list) // n
    chunks = [
        completions_list[
            i * chunk_size : (i + 1) * chunk_size if i < n - 1 else len(completions_list)
        ]
        for i in range(n)
    ]
    ctx = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as ex:
        results = list(
            ex.map(
                compute_rewards_worker,
                chunks,
                [reward_model_path] * n,
                [batch_size] * n,
                [max_seq_length] * n,
                range(n),
                [is_causal_rm] * n,
            )
        )
    merged = [[] for _ in range(4)]
    for result in results:
        for i in range(4):
            merged[i].extend(result[i])
    return tuple(merged)


def score_sharded(
    completions_list,
    model,
    tokenizer,
    batch_size,
    max_seq_length,
    is_causal_rm=False,
):
    """Score completions with a pre-loaded model sharded across GPUs."""
    input_device = next(model.parameters()).device
    all_rewards, all_tokens, all_tokens_len, all_texts = [], [], [], []
    for i in tqdm(range(0, len(completions_list), batch_size), desc="Sharded inference"):
        batch = completions_list[i : i + batch_size]
        tok = tokenizer.apply_chat_template(
            batch,
            return_tensors="pt",
            padding=True,
            return_dict=True,
            truncation=True,
            max_length=max_seq_length,
        )
        for j in range(len(tok["attention_mask"])):
            mask = tok["attention_mask"][j]
            tlen = int(torch.sum(mask).item())
            toks = tok["input_ids"][j][:tlen].tolist()
            all_tokens.append(toks)
            all_tokens_len.append(tlen)
            all_texts.append(tokenizer.decode(toks, skip_special_tokens=False))
        with torch.no_grad():
            attn = tok["attention_mask"].to(input_device)
            out = model(
                input_ids=tok["input_ids"].to(input_device),
                attention_mask=attn,
            )
        scores = _extract_scores(out, attn, is_causal_rm)
        all_rewards.extend(scores.cpu().tolist())
    return all_rewards, all_tokens, all_tokens_len, all_texts


def _symlink(src: Path, dst: Path):
    if not dst.exists() and src.exists():
        dst.symlink_to(src.resolve())


def prepare_prefixed_dirs(
    checkpoints,
    prefix,
    eval_types,
    num_seeds,
    reward_model,
    batch_size,
    max_seq_length,
    num_gpus,
    sharded_model=None,
    sharded_tokenizer=None,
    is_causal_rm=False,
):
    """
    Build prefixed online-eval directories with new reward datasets.
    """
    for eval_type in eval_types:
        for checkpoint in checkpoints:
            for seed in range(num_seeds):
                orig = checkpoint / f"online_evals_{seed}" / eval_type
                new = checkpoint / f"{prefix}online_evals_{seed}" / eval_type

                if not orig.exists():
                    continue
                new.mkdir(parents=True, exist_ok=True)

                _symlink(orig / "completions-dataset", new / "completions-dataset")
                if eval_type == "alpaca_eval":
                    _symlink(orig / "model_outputs.json", new / "model_outputs.json")

                rewards_path = new / "rewards-dataset"
                if rewards_path.exists():
                    import shutil

                    shutil.rmtree(rewards_path)
                logger.info(f"Scoring {orig / 'completions-dataset'}")
                ds = datasets.load_from_disk(str(orig / "completions-dataset"))
                if sharded_model is not None:
                    rewards, tokens, tokens_len, texts = score_sharded(
                        ds["prompt_and_completion"],
                        sharded_model,
                        sharded_tokenizer,
                        batch_size,
                        max_seq_length,
                        is_causal_rm=is_causal_rm,
                    )
                else:
                    rewards, tokens, tokens_len, texts = score_parallel(
                        ds["prompt_and_completion"],
                        reward_model,
                        batch_size,
                        max_seq_length,
                        num_gpus,
                        is_causal_rm=is_causal_rm,
                    )
                rds = (
                    ds.add_column("reward", rewards)
                    .add_column("reward_tokens", tokens)
                    .add_column("reward_tokens_len", tokens_len)
                    .add_column("reward_text", texts)
                )
                rds.save_to_disk(str(rewards_path))
                logger.info(f"Saved {rewards_path}")

                orig_kl = orig / "kl-dataset"
                new_kl = new / "kl-dataset"
                if orig_kl.exists():
                    if new_kl.exists():
                        import shutil

                        shutil.rmtree(new_kl)
                    logger.info(f"Building merged kl-dataset at {new_kl}")
                    kl = datasets.load_from_disk(str(orig_kl))
                    rds = datasets.load_from_disk(str(rewards_path))
                    cols_to_remove = [
                        col for col in REWARD_COLUMNS if col in kl.column_names
                    ]
                    if cols_to_remove:
                        kl = kl.remove_columns(cols_to_remove)
                    for col in REWARD_COLUMNS:
                        kl = kl.add_column(col, rds[col])
                    kl.save_to_disk(str(new_kl))


def discover_training_runs(root: Path, checkpoint_steps=None):
    """
    Discover all training runs under a directory, optionally restricted by step.
    """
    runs = []
    config_paths = sorted(
        root.rglob("checkpoints/*/config/process-0/config-postprocessed.yaml")
    )

    if not config_paths:
        raise ValueError(f"No training runs found under {root}")

    for config_path in config_paths:
        hash_dir = config_path.parent.parent.parent
        training_dir = hash_dir.parent.parent

        with open(config_path) as f:
            saved_config = yaml.safe_load(f)

        checkpoints = sorted(
            (
                d
                for d in hash_dir.iterdir()
                if d.is_dir()
                and d.name.startswith("checkpoint-")
                and (d.name == "checkpoint-0" or not d.is_symlink())
            ),
            key=lambda p: int(re.search(r"\d+", p.name).group()),
        )
        checkpoints = select_checkpoint_paths(
            checkpoints,
            checkpoint_steps,
            context=hash_dir,
        )

        if checkpoints:
            runs.append((training_dir, saved_config, checkpoints))

    return runs


def build_eval_cmd(
    saved_config,
    training_dir,
    prefix,
    eval_type,
    model_name_override=None,
    dataset_name_override=None,
    ref_model_key_override=None,
    checkpoint_steps=None,
):
    """Build the command to call run_online_eval.py with the right overrides."""
    tc = saved_config["training_args"]
    missing = []
    if not model_name_override:
        missing.append("--model")
    if not dataset_name_override:
        missing.append("--dataset")
    if not ref_model_key_override:
        missing.append("--ref_model_key")
    if missing:
        raise ValueError(
            "Phase 2 eval requires explicit rescore overrides: "
            + ", ".join(missing)
        )

    cmd = [
        sys.executable,
        "-m",
        "qrpo.evals.run_online_eval",
        f"dataset={dataset_name_override}",
        f"dataset_args.dataset_path={saved_config['dataset_args']['dataset_path']}",
        (
            "dataset_args.train_split.name="
            f"{saved_config['dataset_args']['train_split']['name']}"
        ),
        (
            "dataset_args.eval_split.name="
            f"{saved_config['dataset_args']['eval_split']['name']}"
        ),
        f"model={model_name_override}",
        f"model_args.model_name_or_path={saved_config['model_args']['model_name_or_path']}",
        f"training_dir={Path(training_dir).resolve()}",
        f"eval_type={eval_type}",
        f"custom_eval_prefix={prefix}",
        f"training_args.loss_type={tc['loss_type']}",
        f"training_args.qrpo_transform_type={tc['qrpo_transform_type']}",
        f"training_args.qrpo_loss_type={tc['qrpo_loss_type']}",
        f"training_args.qrpo_sample_selector={tc['qrpo_sample_selector']}",
        f"training_args.num_ref_rewards={tc['num_ref_rewards']}",
        f"training_args.beta={tc['beta']}",
        f"job_subdir={saved_config['job_subdir']}",
        "outputs_subdir=shared",
        "resuming.resume=False",
        "wandb.mode=disabled",
    ]

    normalized_checkpoint_steps = normalize_checkpoint_steps(checkpoint_steps)
    if normalized_checkpoint_steps:
        joined_steps = ",".join(str(step) for step in normalized_checkpoint_steps)
        cmd.append(f"checkpoint_steps=[{joined_steps}]")

    if "data_dir" in saved_config:
        cmd.append(f"data_dir={saved_config['data_dir']}")

    cmd.append(f"++ref_model_key={ref_model_key_override}")

    if prefix:
        rm_name = prefix.rstrip("_")
        if eval_type == "alpaca_eval":
            cmd.append(f"alpaca_ref_completions_rm={rm_name}")
        elif eval_type == "eval_split":
            cmd.append(f"eval_split_ref_completions_rm={rm_name}")

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Re-score completions with a different reward model, then run eval."
    )
    parser.add_argument(
        "--training_dir",
        required=True,
        help="Single training run dir, or parent dir to discover all runs under",
    )
    parser.add_argument(
        "--reward_model",
        required=True,
        help="HuggingFace model name/path for the new reward model",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Prefix for output directories/files (e.g. 'skywork_')",
    )
    parser.add_argument(
        "--eval_types",
        nargs="+",
        default=["alpaca_eval", "eval_split"],
    )
    parser.add_argument(
        "--checkpoint_steps",
        nargs="+",
        default=None,
        help="Positive checkpoint steps to rescore; omitted means all checkpoints.",
    )
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument(
        "--model",
        default=None,
        help="Explicit Hydra model group for Phase 2 eval, e.g. mistral.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Explicit Hydra dataset group for Phase 2 eval, e.g. magpieair-armorm.",
    )
    parser.add_argument(
        "--ref_model_key",
        default=None,
        help="Explicit Alpaca ref-completion key for Phase 2 LC metrics.",
    )
    parser.add_argument(
        "--skip_phase1",
        action="store_true",
        help="Skip Phase 1 (reward scoring). Useful to run Phase 2 only.",
    )
    parser.add_argument(
        "--skip_phase2",
        action="store_true",
        help="Skip Phase 2 (eval pipeline). Useful to run Phase 1 only.",
    )
    args = parser.parse_args()

    if args.skip_phase1 and args.skip_phase2:
        parser.error("Cannot skip both Phase 1 and Phase 2")
    if not args.skip_phase2:
        missing_eval_args = [
            flag
            for flag, value in (
                ("--model", args.model),
                ("--dataset", args.dataset),
                ("--ref_model_key", args.ref_model_key),
            )
            if not value
        ]
        if missing_eval_args:
            parser.error(
                "Phase 2 eval requires explicit arguments: "
                + ", ".join(missing_eval_args)
            )

    training_runs = discover_training_runs(
        Path(args.training_dir), checkpoint_steps=args.checkpoint_steps
    )
    logger.info(f"Discovered {len(training_runs)} training run(s)")

    if not args.skip_phase1:
        is_causal_rm = _is_causal_lm_reward_model(args.reward_model)
        if is_causal_rm:
            max_causal_bs = 4
            if args.batch_size > max_causal_bs:
                logger.info(
                    f"CausalLM reward model detected, reducing batch_size "
                    f"from {args.batch_size} to {max_causal_bs}"
                )
                args.batch_size = max_causal_bs
            logger.info("Detected CausalLM reward model")

        sharded_model, sharded_tokenizer = None, None
        if _model_needs_sharding(args.reward_model):
            logger.info("Model too large for single GPU, loading with device_map='auto'")
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            model_cls = (
                AutoModelForCausalLM
                if is_causal_rm
                else AutoModelForSequenceClassification
            )
            sharded_model = model_cls.from_pretrained(
                args.reward_model,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            sharded_tokenizer = AutoTokenizer.from_pretrained(
                args.reward_model, use_fast=True
            )
            if sharded_tokenizer.pad_token is None:
                sharded_tokenizer.pad_token = sharded_tokenizer.eos_token
            logger.info(
                f"Model loaded across devices: {set(sharded_model.hf_device_map.values())}"
            )
        else:
            logger.info("Model fits on single GPU, using data-parallel scoring")

        for run_idx, (training_dir, _saved_config, checkpoints) in enumerate(training_runs):
            logger.info(
                f"[{run_idx + 1}/{len(training_runs)}] {training_dir.name} "
                f"({len(checkpoints)} checkpoints)"
            )
            prepare_prefixed_dirs(
                checkpoints,
                args.prefix,
                args.eval_types,
                args.num_seeds,
                args.reward_model,
                args.batch_size,
                args.max_seq_length,
                args.num_gpus,
                sharded_model=sharded_model,
                sharded_tokenizer=sharded_tokenizer,
                is_causal_rm=is_causal_rm,
            )

        if sharded_model is not None:
            del sharded_model
            del sharded_tokenizer
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Freed sharded model GPU memory before eval pipeline")

    if not args.skip_phase2:
        for _run_idx, (training_dir, saved_config, _checkpoints) in enumerate(training_runs):
            for eval_type in args.eval_types:
                cmd = build_eval_cmd(
                    saved_config,
                    str(training_dir),
                    args.prefix,
                    eval_type,
                    model_name_override=args.model,
                    dataset_name_override=args.dataset,
                    ref_model_key_override=args.ref_model_key,
                    checkpoint_steps=args.checkpoint_steps,
                )
                logger.info(f"Running eval pipeline: {training_dir.name} / {eval_type}")
                subprocess.run(cmd, check=True)

    logger.info(f"Done! Processed {len(training_runs)} training run(s).")


if __name__ == "__main__":
    main()
