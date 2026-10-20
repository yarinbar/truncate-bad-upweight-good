from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from datasets import load_from_disk
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MERGED_DIR = REPO_ROOT / "data/shared/datasets-with-ref-rewards/merged"
DEFAULT_ALPACA_REF_COMPLETIONS_DIR = REPO_ROOT / "data/shared/datasets/alpaca-ref-completions"
DEFAULT_EVAL_RMS = ("skywork_v2_llama", "skywork_v2_qwen")

MODEL_ALIASES = {
    "llama": "llama",
    "mistral": "mistral",
    "mistralsft": "mistral-sft",
}

DATASET_ALIASES = {
    "magpie": "magpie",
    "magpieair": "magpie",
    "ultrafeedback": "ultrafeedback",
    "ultrafeed": "ultrafeedback",
}

DISPLAY_DATASET_NAMES = {
    "magpie": "Magpie",
    "ultrafeedback": "UltraFeedback",
}

DISPLAY_MODEL_NAMES = {
    "llama": "Llama",
    "mistral": "Mistral",
    "mistral-sft": "Mistral SFT",
}

VARIANT_ALIASES = {
    "best": "best",
    "random": "random",
}

EVAL_SOURCE_ALIASES = {
    "merged": "merged",
    "evalsplit": "merged",
    "evalsplitrewards": "merged",
    "alpaca": "alpaca",
    "alpacaeval": "alpaca",
}

_MERGED_DATASET_RE = re.compile(
    r"^(?P<model>[a-z0-9_]+)-(?P<tuning>nosft|sft)-(?P<dataset>[a-z0-9_]+)-armorm-temp1-ref50-"
    r"offpolicy2(?P<variant>best|random)-armorm$"
)

_ALPACA_ALIGNMENT_COLUMNS = ("instruction", "ref_completions")

@dataclass(frozen=True)
class StaticRewardSource:
    model: str
    dataset: str
    variant: str
    path: Path
    eval_source: str = "merged"
    split: Optional[str] = None
    armorm_path: Optional[Path] = None
    armorm_split: Optional[str] = None
    eval_path: Optional[Path] = None
    eval_split: Optional[str] = None

    @property
    def model_display(self) -> str:
        return DISPLAY_MODEL_NAMES.get(self.model, self.model.title())

    @property
    def dataset_display(self) -> str:
        return DISPLAY_DATASET_NAMES.get(self.dataset, self.dataset.title())

    @property
    def selection_label(self) -> str:
        return f"{self.model_display} + {self.dataset_display}"

    @property
    def uses_mixed_sources(self) -> bool:
        return (
            self.eval_source == "merged"
            and self.armorm_path is not None
            and self.eval_path is not None
            and (self.armorm_path != self.eval_path or self.armorm_split != self.eval_split)
        )

    @property
    def load_summary(self) -> str:
        parts = [self.variant]
        if self.split:
            parts.append(f"split={self.split}")

        if self.armorm_path is not None and self.eval_path is not None:
            if self.uses_mixed_sources:
                parts.append(f"armorm={self.armorm_path.name}")
                parts.append(f"eval={self.eval_path.name}")
            elif self.eval_source == "merged":
                parts.append(f"source={self.eval_path.name}")
            else:
                parts.append(f"{self.eval_source}={self.eval_path.name}")

        return ", ".join(parts)


def _canonicalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def canonicalize_model(value: str) -> str:
    key = _canonicalize_key(value)
    if key not in MODEL_ALIASES:
        raise ValueError(f"Unknown model '{value}'.")
    return MODEL_ALIASES[key]


def canonicalize_dataset(value: str) -> str:
    key = _canonicalize_key(value)
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unknown dataset '{value}'.")
    return DATASET_ALIASES[key]


def canonicalize_variant(value: str) -> str:
    key = _canonicalize_key(value)
    if key not in VARIANT_ALIASES:
        raise ValueError(f"Unknown variant '{value}'.")
    return VARIANT_ALIASES[key]


def canonicalize_eval_source(value: str) -> str:
    key = _canonicalize_key(value)
    if key not in EVAL_SOURCE_ALIASES:
        raise ValueError(f"Unknown eval source '{value}'.")
    return EVAL_SOURCE_ALIASES[key]


def _variant_rank(variant: str) -> int:
    return {"best": 0, "random": 1}.get(variant, 99)


def _reward_model_rank(rm: str) -> Tuple[int, str]:
    order = ("armorm",) + DEFAULT_EVAL_RMS
    try:
        return order.index(rm), rm
    except ValueError:
        return len(order), rm


def _parse_source(path: Path) -> Optional[StaticRewardSource]:
    match = _MERGED_DATASET_RE.match(path.name)
    if not match:
        return None

    tuning = match.group("tuning")
    model_label = match.group("model") if tuning == "nosft" else f"{match.group('model')}-{tuning}"
    model = canonicalize_model(model_label)
    dataset = canonicalize_dataset(match.group("dataset"))
    variant = match.group("variant")
    return StaticRewardSource(model=model, dataset=dataset, variant=variant, path=path)


def discover_static_reward_sources(
    base_dir: Path = DEFAULT_MERGED_DIR,
    variant: Optional[str] = None,
) -> Dict[Tuple[str, str], StaticRewardSource]:
    sources: Dict[Tuple[str, str], StaticRewardSource] = {}
    variant_key = canonicalize_variant(variant) if variant is not None else None

    if not base_dir.exists():
        return sources

    for path in sorted(base_dir.iterdir()):
        if not path.is_dir():
            continue
        source = _parse_source(path)
        if source is None:
            continue
        if variant_key is not None and source.variant != variant_key:
            continue

        key = (source.model, source.dataset)
        current = sources.get(key)
        if current is None or _variant_rank(source.variant) < _variant_rank(current.variant):
            sources[key] = source

    return sources


def discover_static_reward_variants(
    base_dir: Path = DEFAULT_MERGED_DIR,
) -> Dict[Tuple[str, str], Sequence[str]]:
    variants: Dict[Tuple[str, str], list[str]] = {}

    if not base_dir.exists():
        return {}

    for path in sorted(base_dir.iterdir()):
        if not path.is_dir():
            continue
        source = _parse_source(path)
        if source is None:
            continue

        variants.setdefault((source.model, source.dataset), []).append(source.variant)

    return {
        key: tuple(sorted(set(source_variants), key=_variant_rank))
        for key, source_variants in sorted(variants.items())
    }


def datasets_by_model(
    sources: Optional[Mapping[Tuple[str, str], StaticRewardSource]] = None,
) -> Dict[str, Sequence[str]]:
    if sources is None:
        sources = discover_static_reward_sources()

    options: Dict[str, list[str]] = {}
    for model, dataset in sources:
        options.setdefault(model, []).append(dataset)

    return {model: tuple(sorted(datasets)) for model, datasets in sorted(options.items())}


def resolve_static_reward_source(
    model: str,
    dataset: str,
    base_dir: Path = DEFAULT_MERGED_DIR,
    variant: Optional[str] = None,
) -> StaticRewardSource:
    model_key = canonicalize_model(model)
    dataset_key = canonicalize_dataset(dataset)
    variant_key = canonicalize_variant(variant) if variant is not None else None
    sources = discover_static_reward_sources(base_dir=base_dir, variant=variant_key)
    key = (model_key, dataset_key)

    if key not in sources:
        variants = discover_static_reward_variants(base_dir=base_dir).get(key, ())
        variant_suffix = f" with variant {variant_key}" if variant_key is not None else ""
        available_variants = ", ".join(variants) or "none"
        available = ", ".join(
            f"{source.model}+{source.dataset}" for source in sorted(sources.values(), key=lambda s: (s.model, s.dataset))
        )
        raise FileNotFoundError(
            f"No merged ref-reward dataset found for {model_key}+{dataset_key}{variant_suffix}. "
            f"Available variants for this selection: {available_variants}. "
            f"Available combinations: {available or 'none'}."
        )

    return sources[key]


def _load_primary_split(source: StaticRewardSource):
    dataset_dict = load_from_disk(str(source.path))
    for split in ("test", "test_prefs"):
        if split in dataset_dict:
            return dataset_dict[split], replace(
                source,
                split=split,
                armorm_path=source.path,
                armorm_split=split,
                eval_path=source.path,
                eval_split=split,
            )
    raise KeyError(f"No supported evaluation split found in {source.path}.")


def _as_dataset(dataset_or_dict, source_path: Path):
    if hasattr(dataset_or_dict, "column_names"):
        return dataset_or_dict
    if "eval" in dataset_or_dict:
        return dataset_or_dict["eval"]
    if "test" in dataset_or_dict:
        return dataset_or_dict["test"]
    split_names = list(dataset_or_dict)
    if not split_names:
        raise KeyError(f"No splits found in {source_path}.")
    return dataset_or_dict[split_names[0]]


def _load_eval_rm_dataset(source: StaticRewardSource, rm: str):
    eval_root = source.eval_path or source.path
    eval_path = eval_root / "eval-split-rewards" / rm
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval RM dataset at {eval_path}.")
    return load_from_disk(str(eval_path))


def _discover_merged_eval_rms(source: StaticRewardSource, include_train_rm: bool = False) -> Tuple[str, ...]:
    eval_root = source.path / "eval-split-rewards"
    rms = []
    if eval_root.exists():
        rms = [path.name for path in eval_root.iterdir() if path.is_dir()]
    if include_train_rm:
        rms.append("armorm")
    return tuple(sorted(set(rms), key=_reward_model_rank))


def _discover_alpaca_eval_rms(
    model: str,
    alpaca_dir: Path = DEFAULT_ALPACA_REF_COMPLETIONS_DIR,
    include_train_rm: bool = False,
) -> Tuple[str, ...]:
    model_dir = alpaca_dir / canonicalize_model(model)
    if not model_dir.exists():
        return ()

    rms = [
        path.name
        for path in model_dir.iterdir()
        if path.is_dir() and path.name != "raw-completions"
    ]
    if not include_train_rm:
        rms = [rm for rm in rms if rm != "armorm"]
    return tuple(sorted(set(rms), key=_reward_model_rank))


def discover_available_eval_rms(
    model: str,
    dataset: str,
    base_dir: Path = DEFAULT_MERGED_DIR,
    variant: Optional[str] = None,
    eval_source: str = "merged",
    alpaca_dir: Path = DEFAULT_ALPACA_REF_COMPLETIONS_DIR,
    include_train_rm: bool = False,
) -> Tuple[str, ...]:
    source = resolve_static_reward_source(model=model, dataset=dataset, base_dir=base_dir, variant=variant)
    eval_source_key = canonicalize_eval_source(eval_source)
    if eval_source_key == "merged":
        return _discover_merged_eval_rms(source, include_train_rm=include_train_rm)
    if eval_source_key == "alpaca":
        return _discover_alpaca_eval_rms(source.model, alpaca_dir=alpaca_dir, include_train_rm=include_train_rm)
    raise ValueError(f"Unsupported eval source '{eval_source}'.")


def _select_eval_rms(
    eval_rms: Optional[Iterable[str]],
    available_rms: Sequence[str],
    source_path: Path,
) -> Tuple[str, ...]:
    available = set(available_rms)
    if eval_rms is None:
        selected = [rm for rm in available_rms if rm != "armorm"]
    else:
        requested_rms = tuple(eval_rms)
        selected = [rm for rm in requested_rms if rm in available]
        if not selected and requested_rms:
            requested = ", ".join(requested_rms)
            available_label = ", ".join(available_rms) or "none"
            raise FileNotFoundError(
                f"No requested eval RM datasets were found under {source_path}. "
                f"Requested: {requested}; available: {available_label}."
            )
    return tuple(selected)


def _load_alpaca_rm_dataset(model_dir: Path, rm: str):
    rm_path = model_dir / rm
    if not rm_path.exists():
        raise FileNotFoundError(f"Missing AlpacaEval RM dataset at {rm_path}.")
    return _as_dataset(load_from_disk(str(rm_path)), rm_path)


def _assert_alpaca_alignment(base_dataset, base_path: Path, eval_dataset, eval_path: Path) -> None:
    if len(base_dataset) != len(eval_dataset):
        raise ValueError(
            f"Alpaca reward dataset {eval_path} is not aligned with {base_path}: "
            f"expected {len(base_dataset)} rows, got {len(eval_dataset)}."
        )

    for column in _ALPACA_ALIGNMENT_COLUMNS:
        if column not in base_dataset.column_names or column not in eval_dataset.column_names:
            raise KeyError(f"Alignment column '{column}' is missing in {base_path} or {eval_path}.")
        if list(base_dataset[column]) != list(eval_dataset[column]):
            raise ValueError(
                f"Alpaca reward dataset {eval_path} is not aligned with {base_path} on column '{column}'."
            )


def _ref_rewards_array(dataset, source_path: Path) -> np.ndarray:
    if "ref_rewards" not in dataset.column_names:
        raise KeyError(f"'ref_rewards' column is missing in {source_path}.")

    rewards = np.asarray(dataset["ref_rewards"], dtype=float)
    if rewards.ndim != 2:
        raise ValueError(f"Expected a 2D ref_rewards array in {source_path}, got shape {rewards.shape}.")
    return rewards


def load_static_reward_arrays(
    model: str,
    dataset: str,
    eval_rms: Optional[Iterable[str]] = DEFAULT_EVAL_RMS,
    base_dir: Path = DEFAULT_MERGED_DIR,
    variant: Optional[str] = None,
    eval_source: str = "merged",
    alpaca_dir: Path = DEFAULT_ALPACA_REF_COMPLETIONS_DIR,
):
    source = resolve_static_reward_source(model=model, dataset=dataset, base_dir=base_dir, variant=variant)
    eval_source_key = canonicalize_eval_source(eval_source)

    if eval_source_key == "merged":
        train_dataset, source = _load_primary_split(source)
        available_rms = _discover_merged_eval_rms(source)
        selected_eval_rms = _select_eval_rms(eval_rms, available_rms, source.path / "eval-split-rewards")

        rewards = {"armorm": _ref_rewards_array(train_dataset, source.path)}
        for rm in selected_eval_rms:
            eval_root = source.eval_path or source.path
            rewards[rm] = _ref_rewards_array(_load_eval_rm_dataset(source, rm), eval_root / "eval-split-rewards" / rm)
    elif eval_source_key == "alpaca":
        alpaca_model_dir = alpaca_dir / source.model
        available_rms = _discover_alpaca_eval_rms(source.model, alpaca_dir=alpaca_dir)
        selected_eval_rms = _select_eval_rms(eval_rms, available_rms, alpaca_model_dir)

        armorm_path = alpaca_model_dir / "armorm"
        train_dataset = _load_alpaca_rm_dataset(alpaca_model_dir, "armorm")
        source = replace(
            source,
            eval_source="alpaca",
            split="eval",
            armorm_path=armorm_path,
            armorm_split="eval",
            eval_path=alpaca_model_dir,
            eval_split="eval",
        )

        rewards = {"armorm": _ref_rewards_array(train_dataset, armorm_path)}
        for rm in selected_eval_rms:
            eval_path = alpaca_model_dir / rm
            eval_dataset = _load_alpaca_rm_dataset(alpaca_model_dir, rm)
            _assert_alpaca_alignment(train_dataset, armorm_path, eval_dataset, eval_path)
            rewards[rm] = _ref_rewards_array(eval_dataset, eval_path)
    else:
        raise ValueError(f"Unsupported eval source '{eval_source}'.")

    expected_shape = rewards["armorm"].shape
    for rm, values in rewards.items():
        if values.shape != expected_shape:
            raise ValueError(
                f"Reward shape mismatch for {rm}: expected {expected_shape}, got {values.shape}."
            )

    return rewards, source
