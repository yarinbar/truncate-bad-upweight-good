#!/usr/bin/env python3
"""
Generate reference completions for alpaca eval prompts and score them with reward models.

Creates the reference data needed for length-controlled (LC) reward metrics
during alpaca eval evaluation in run_online_eval.py.

Two phases:
  Phase 1 (generation): Generate N completions per alpaca prompt from the base/SFT model.
  Phase 2 (scoring):    Score completions with a reward model and save per-RM dataset.

If completions already exist, Phase 1 is skipped automatically.
If the RM dataset already exists, Phase 2 is skipped (use --overwrite to force).
"""

import argparse
import json
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import datasets
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
SUPPORTED_MODELS = ("llama", "mistral")
HH_ALPACA_PROMPT_TEMPLATE = "\n\nHuman: {instruction}\n\nAssistant:"


def resolve_gpu_ids(num_gpus, gpu_ids=None):
    """Resolve worker GPU ids, honoring an explicit comma-separated GPU_IDS list."""
    if gpu_ids is None:
        gpu_ids = os.environ.get("GPU_IDS", "")

    if str(gpu_ids).strip():
        resolved = [gpu.strip() for gpu in str(gpu_ids).split(",") if gpu.strip()]
        if not resolved:
            raise ValueError(f"Invalid GPU id list: {gpu_ids!r}")
        return resolved[:num_gpus]

    return [str(i) for i in range(num_gpus)]


def _has_chat_template(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def _message_content(message) -> str:
    if isinstance(message, dict):
        return str(message["content"])
    return str(message)


def build_generation_prompt(instruction, tokenizer, prompt_template):
    """Render an Alpaca instruction for the reference model."""
    if prompt_template == "chat_template":
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
    if prompt_template == "raw":
        return instruction
    if prompt_template == "hh":
        return HH_ALPACA_PROMPT_TEMPLATE.format(instruction=instruction)
    raise ValueError(f"Unsupported prompt template: {prompt_template}")


def tokenize_reward_batch(tokenizer, batch, max_seq_length):
    """Tokenize reward-model inputs, falling back for non-chat-template RMs."""
    if _has_chat_template(tokenizer):
        return tokenizer.apply_chat_template(
            batch,
            return_tensors="pt",
            padding=True,
            return_dict=True,
            truncation=True,
            max_length=max_seq_length,
        )

    prompts = [_message_content(item[0]) for item in batch]
    completions = [_message_content(item[1]) for item in batch]
    return tokenizer(
        prompts,
        completions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
    )


def generate_completions_worker(
    base_model_path,
    prompts_chunk,
    n_completions,
    max_tokens,
    temperature,
    top_p,
    torch_dtype,
    gpu_id,
    seed,
):
    """Generate N completions per prompt on a single GPU using vLLM."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        n=n_completions,
        max_tokens=max_tokens,
        seed=seed,
    )

    llm = LLM(
        base_model_path,
        dtype=torch_dtype,
        gpu_memory_utilization=0.95,
        max_model_len=4096,
    )
    outputs = llm.generate(prompts_chunk, sampling_params)

    return [[c.text.strip() for c in output.outputs] for output in outputs]


def generate_completions(
    prompts,
    base_model_path,
    n_completions,
    max_tokens,
    temperature,
    top_p,
    torch_dtype,
    num_gpus,
    seed,
):
    """Generate completions in parallel across GPUs."""
    gpu_ids = resolve_gpu_ids(num_gpus)
    n = min(len(gpu_ids), len(prompts))
    chunk_size = len(prompts) // n
    chunks = [
        prompts[
            i * chunk_size : (i + 1) * chunk_size if i < n - 1 else len(prompts)
        ]
        for i in range(n)
    ]

    logger.info(
        f"Generating {n_completions} completions per prompt across {n} GPUs "
        f"{gpu_ids[:n]} ({len(prompts)} prompts total)"
    )

    ctx = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as ex:
        results = list(
            ex.map(
                generate_completions_worker,
                [base_model_path] * n,
                chunks,
                [n_completions] * n,
                [max_tokens] * n,
                [temperature] * n,
                [top_p] * n,
                [torch_dtype] * n,
                gpu_ids[:n],
                [seed] * n,
            )
        )

    all_completions = []
    for chunk_result in results:
        all_completions.extend(chunk_result)

    return all_completions


def _extract_scores(out, attention_mask, is_causal_rm):
    """Extract reward scores from model output."""
    if is_causal_rm:
        last_pos = attention_mask.sum(dim=1) - 1
        return out.logits[torch.arange(out.logits.size(0)), last_pos, 0]
    scores = out.score if hasattr(out, "score") else out.logits
    return scores.squeeze(-1)


def _is_causal_lm_reward_model(model_path):
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        architectures = getattr(config, "architectures", [])
        return any("CausalLM" in arch for arch in architectures)
    except Exception:
        return False


def _model_needs_sharding(model_path):
    import subprocess as sp

    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        h = getattr(config, "hidden_size", 0)
        n = getattr(config, "num_hidden_layers", 0)
        v = getattr(config, "vocab_size", 0)
        inter = getattr(config, "intermediate_size", h * 4)
        est_params = n * (4 * h * h + 3 * h * inter) + v * h * 2
        est_gb = est_params * 2 / (1024**3)
        try:
            out = sp.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            gpu_mem_gb = int(out.stdout.strip().split("\n")[0]) / 1024
        except Exception:
            gpu_mem_gb = 80.0
        logger.info(
            f"Model size estimate: {est_gb:.1f} GB (bf16), GPU memory: {gpu_mem_gb:.1f} GB"
        )
        return est_gb > gpu_mem_gb * 0.85
    except Exception as exc:
        logger.warning(f"Could not estimate model size: {exc}")
        return False


def score_completions_worker(
    chunk_data,
    reward_model_path,
    batch_size,
    max_seq_length,
    gpu_id,
    is_causal_rm=False,
):
    """Score flattened chat completions on a single GPU."""
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

    all_rewards = []
    all_tokens_len = []
    for i in tqdm(range(0, len(chunk_data), batch_size), desc=f"GPU {gpu_id}"):
        batch = chunk_data[i : i + batch_size]
        tok = tokenize_reward_batch(tokenizer, batch, max_seq_length)
        for j in range(len(tok["attention_mask"])):
            tlen = int(torch.sum(tok["attention_mask"][j]).item())
            all_tokens_len.append(tlen)
        with torch.no_grad():
            attn = tok["attention_mask"].to("cuda:0")
            out = model(
                input_ids=tok["input_ids"].to("cuda:0"),
                attention_mask=attn,
            )
        scores = _extract_scores(out, attn, is_causal_rm)
        all_rewards.extend(scores.cpu().tolist())

    return all_rewards, all_tokens_len


def score_ref_completions(
    instructions,
    completions_per_prompt,
    rm_path,
    batch_size,
    max_seq_length,
    num_gpus,
):
    """Score N completions per prompt with a reward model."""
    n_completions = len(completions_per_prompt[0])
    n_prompts = len(instructions)

    all_chats = []
    for instruction, completions in zip(instructions, completions_per_prompt):
        for completion in completions:
            all_chats.append(
                [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": completion},
                ]
            )

    logger.info(
        f"Scoring {len(all_chats)} chat pairs "
        f"({n_prompts} prompts x {n_completions} completions)"
    )

    is_causal_rm = _is_causal_lm_reward_model(rm_path)
    if is_causal_rm:
        max_causal_bs = 4
        if batch_size > max_causal_bs:
            logger.info(
                f"CausalLM RM detected, reducing batch_size to {max_causal_bs}"
            )
            batch_size = max_causal_bs

    if _model_needs_sharding(rm_path):
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
        model = model_cls.from_pretrained(
            rm_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(rm_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        input_device = next(model.parameters()).device
        all_rewards = []
        all_tokens_len = []
        for i in tqdm(range(0, len(all_chats), batch_size), desc="Sharded inference"):
            batch = all_chats[i : i + batch_size]
            tok = tokenize_reward_batch(tokenizer, batch, max_seq_length)
            for j in range(len(tok["attention_mask"])):
                tlen = int(torch.sum(tok["attention_mask"][j]).item())
                all_tokens_len.append(tlen)
            with torch.no_grad():
                attn = tok["attention_mask"].to(input_device)
                out = model(
                    input_ids=tok["input_ids"].to(input_device),
                    attention_mask=attn,
                )
            scores = _extract_scores(out, attn, is_causal_rm)
            all_rewards.extend(scores.cpu().tolist())

        del model, tokenizer
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    else:
        gpu_ids = resolve_gpu_ids(num_gpus)
        n = min(len(gpu_ids), len(all_chats))
        chunk_size = len(all_chats) // n
        chunks = [
            all_chats[
                i * chunk_size : (i + 1) * chunk_size if i < n - 1 else len(all_chats)
            ]
            for i in range(n)
        ]

        ctx = multiprocessing.get_context("forkserver")
        with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as ex:
            results = list(
                ex.map(
                    score_completions_worker,
                    chunks,
                    [rm_path] * n,
                    [batch_size] * n,
                    [max_seq_length] * n,
                    gpu_ids[:n],
                    [is_causal_rm] * n,
                )
            )

        all_rewards = []
        all_tokens_len = []
        for rewards, tokens_len in results:
            all_rewards.extend(rewards)
            all_tokens_len.extend(tokens_len)

    ref_rewards = [
        all_rewards[i * n_completions : (i + 1) * n_completions]
        for i in range(n_prompts)
    ]
    ref_tokens_len = [
        all_tokens_len[i * n_completions : (i + 1) * n_completions]
        for i in range(n_prompts)
    ]

    return ref_rewards, ref_tokens_len


def main():
    parser = argparse.ArgumentParser(
        description="Generate and score reference completions for alpaca eval LC metrics.",
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default=None,
        help=(
            "Convenience model selector. Sets --model_key automatically and, if "
            "--base_model is omitted, defaults it to {data_dir}/shared/models/<model>."
        ),
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Path to base/SFT model for generation (required unless --skip_generation)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Root data directory (default: data)",
    )
    parser.add_argument(
        "--model_key",
        type=str,
        default="",
        help="Model subdirectory for output (e.g. 'mistral'). Empty means output to root.",
    )
    parser.add_argument(
        "--n_completions",
        type=int,
        default=16,
        help="Number of completions per prompt (default: 16)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Max tokens per completion (default: 2048)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling (default: 1.0)",
    )
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--prompt-template",
        choices=("chat_template", "raw", "hh"),
        default="chat_template",
        help=(
            "Prompt format for generation. Use 'hh' for Pythia HH SFT models, "
            "'chat_template' for chat-tokenizer models, or 'raw' for bare instructions."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--rm_name",
        type=str,
        default=None,
        help="Short name for the reward model (e.g., armorm, skywork)",
    )
    parser.add_argument(
        "--rm_path",
        type=str,
        default=None,
        help="HuggingFace model path for the reward model",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip Phase 1 (completions must already exist)",
    )
    parser.add_argument(
        "--skip_scoring",
        action="store_true",
        help="Skip Phase 2 (only generate completions)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing datasets",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.model:
        if args.model_key and args.model_key != args.model:
            parser.error("--model and --model_key must match when both are provided")
        if not args.model_key:
            args.model_key = args.model
        if args.base_model is None:
            args.base_model = str(data_dir / "shared/models" / args.model)

    output_root = data_dir / "shared/datasets/alpaca-ref-completions"
    if args.model_key:
        output_root = output_root / args.model_key
    completions_path = output_root / "raw-completions"

    if not args.skip_generation:
        if args.base_model is None:
            parser.error(
                "--base_model is required for generation (or use --skip_generation)"
            )

        if completions_path.exists() and not args.overwrite:
            logger.info(
                f"Completions already exist at {completions_path}, skipping generation. "
                f"Use --overwrite to regenerate."
            )
        else:
            alpaca_path = data_dir / "shared/datasets/alpaca"
            alpaca_dataset = datasets.load_from_disk(str(alpaca_path))["eval"]
            logger.info(f"Loaded {len(alpaca_dataset)} alpaca eval prompts")

            tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
                args.base_model
            )
            prompts = [
                build_generation_prompt(
                    row["instruction"],
                    tokenizer=tokenizer,
                    prompt_template=args.prompt_template,
                )
                for row in alpaca_dataset
            ]

            completions_per_prompt = generate_completions(
                prompts,
                args.base_model,
                args.n_completions,
                args.max_tokens,
                args.temperature,
                args.top_p,
                args.torch_dtype,
                args.num_gpus,
                args.seed,
            )

            ref_completions = [
                [
                    {"role": "user", "content": row["instruction"]},
                    {"role": "assistant", "content": json.dumps(completions)},
                ]
                for row, completions in zip(alpaca_dataset, completions_per_prompt)
            ]

            completions_ds = alpaca_dataset.add_column(
                "ref_completions", ref_completions
            )
            completions_ds.save_to_disk(str(completions_path))
            logger.info(f"Saved raw completions to {completions_path}")

    if not args.skip_scoring:
        if args.rm_name is None or args.rm_path is None:
            parser.error(
                "--rm_name and --rm_path are required for scoring "
                "(or use --skip_scoring)"
            )

        rm_output_path = output_root / args.rm_name
        if rm_output_path.exists() and not args.overwrite:
            logger.info(
                f"RM dataset already exists at {rm_output_path}, skipping. "
                f"Use --overwrite to rescore."
            )
        else:
            if not completions_path.exists():
                raise FileNotFoundError(
                    f"Raw completions not found at {completions_path}. "
                    f"Run without --skip_generation first."
                )
            completions_ds = datasets.load_from_disk(str(completions_path))
            logger.info(f"Loaded {len(completions_ds)} prompts with completions")

            instructions = completions_ds["instruction"]
            completions_per_prompt = [
                json.loads(row[1]["content"]) for row in completions_ds["ref_completions"]
            ]

            ref_rewards, ref_tokens_len = score_ref_completions(
                instructions,
                completions_per_prompt,
                args.rm_path,
                args.batch_size,
                args.max_seq_length,
                args.num_gpus,
            )

            rm_ds = completions_ds.add_column("ref_rewards", ref_rewards).add_column(
                "ref_completions_reward_tokens_len", ref_tokens_len
            )

            if rm_output_path.exists():
                import shutil

                shutil.rmtree(rm_output_path)

            rm_ds.save_to_disk(str(rm_output_path))
            logger.info(f"Saved scored dataset to {rm_output_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
