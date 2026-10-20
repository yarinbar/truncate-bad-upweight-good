from __future__ import annotations

import importlib.util
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency for runtime probing
    yaml = None


_NAME_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"gpt[_-]?4o[_-]?mini", re.IGNORECASE), "gpt-4o-mini"),
    (re.compile(r"gpt[_-]?4o", re.IGNORECASE), "gpt-4o"),
    (re.compile(r"gpt[_-]?4(?:[_-]?turbo|turbo)", re.IGNORECASE), "gpt-4-turbo"),
    (re.compile(r"gpt[_-]?4[_-]?1(?!\d)[_-]?mini", re.IGNORECASE), "gpt-4.1-mini"),
    (re.compile(r"gpt[_-]?4[_-]?1(?!\d)[_-]?nano", re.IGNORECASE), "gpt-4.1-nano"),
    (re.compile(r"gpt[_-]?4[_-]?1(?!\d)", re.IGNORECASE), "gpt-4.1"),
]

_MODEL_VALUE_PATTERN = re.compile(
    r"\b(?:gpt[-_0-9.a-z]+|o[1-9](?:[-_][0-9a-z]+)?|claude[-_0-9.a-z]+|gemini[-_0-9.a-z]+)\b",
    re.IGNORECASE,
)
REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_EVALUATORS_CONFIG_DIR = REPO_ROOT / "alpaca_eval_configs"


def _normalize_model_name(candidate: str) -> str:
    candidate = str(candidate).strip()
    for pattern, normalized in _NAME_PATTERNS:
        if pattern.search(candidate):
            return normalized
    return candidate


def _infer_judge_model_from_name(judge_config: str) -> Optional[str]:
    for pattern, normalized in _NAME_PATTERNS:
        if pattern.search(judge_config):
            return normalized
    return None


def _iter_candidate_config_files(package_root: Path, judge_config: str) -> Iterable[Path]:
    exact_stems = {judge_config, judge_config.lower()}
    for path in package_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".yaml", ".yml", ".json", ".txt"}:
            continue
        if path.stem in exact_stems or judge_config in path.name:
            yield path


def _extract_model_like_value(payload: Any) -> Optional[str]:
    if isinstance(payload, dict):
        for key in ("model", "model_name", "engine", "completions_name"):
            value = payload.get(key)
            if isinstance(value, str) and _MODEL_VALUE_PATTERN.search(value):
                return _normalize_model_name(value)
        for value in payload.values():
            found = _extract_model_like_value(value)
            if found:
                return found
        return None

    if isinstance(payload, list):
        for item in payload:
            found = _extract_model_like_value(item)
            if found:
                return found
        return None

    if isinstance(payload, str) and _MODEL_VALUE_PATTERN.search(payload):
        return _normalize_model_name(payload)

    return None


def _load_config_payload(path: Path) -> Any:
    if path.is_dir():
        path = path / "configs.yaml"
    suffix = path.suffix.lower()
    with open(path, "r") as f:
        if suffix == ".json":
            return json.load(f)
        if suffix in {".yaml", ".yml"} and yaml is not None:
            return yaml.safe_load(f)
        return f.read()


def resolve_local_judge_config(judge_config: str | Path) -> Optional[Path]:
    candidate = Path(judge_config).expanduser()
    if candidate.exists():
        return candidate.resolve()

    local_candidate = LOCAL_EVALUATORS_CONFIG_DIR / str(judge_config)
    if local_candidate.exists():
        return local_candidate.resolve()

    return None


def resolve_judge_config(judge_config: str | Path) -> str:
    local = resolve_local_judge_config(judge_config)
    if local is not None:
        return str(local)
    return str(judge_config)


def resolve_installed_judge_config(judge_config: str | Path) -> Optional[Path]:
    spec = importlib.util.find_spec("alpaca_eval")
    if spec is None or not spec.submodule_search_locations:
        return None

    package_root = Path(spec.submodule_search_locations[0])
    candidate = package_root / "evaluators_configs" / str(judge_config)
    if candidate.exists():
        return candidate.resolve()
    return None


def list_available_judge_configs() -> list[str]:
    available = set()

    if LOCAL_EVALUATORS_CONFIG_DIR.exists():
        for child in LOCAL_EVALUATORS_CONFIG_DIR.iterdir():
            if child.is_dir():
                available.add(child.name)

    spec = importlib.util.find_spec("alpaca_eval")
    if spec is not None and spec.submodule_search_locations:
        installed_root = Path(spec.submodule_search_locations[0]) / "evaluators_configs"
        if installed_root.exists():
            for child in installed_root.iterdir():
                available.add(child.name)

    return sorted(available)


def ensure_judge_config(judge_config: str | Path) -> str:
    local = resolve_local_judge_config(judge_config)
    if local is not None:
        return str(local)

    installed = resolve_installed_judge_config(judge_config)
    if installed is not None:
        return str(judge_config)

    available = list_available_judge_configs()
    preview = ", ".join(available[:12])
    if len(available) > 12:
        preview += ", ..."
    raise FileNotFoundError(
        f"Unknown AlpacaEval judge config: {judge_config}. "
        f"Available configs include: {preview}"
    )


@lru_cache(maxsize=None)
def _resolve_judge_model_from_alpaca_eval(judge_config: str) -> Optional[str]:
    installed = resolve_installed_judge_config(judge_config)
    if installed is not None:
        try:
            payload = _load_config_payload(installed)
        except Exception:
            payload = None
        if payload is not None:
            model_name = _extract_model_like_value(payload)
            if model_name:
                return model_name

    spec = importlib.util.find_spec("alpaca_eval")
    if spec is None or not spec.submodule_search_locations:
        return None

    package_root = Path(spec.submodule_search_locations[0])
    for config_path in _iter_candidate_config_files(package_root, judge_config):
        try:
            payload = _load_config_payload(config_path)
        except Exception:
            continue
        model_name = _extract_model_like_value(payload)
        if model_name:
            return model_name
    return None


def infer_judge_model(judge_config: str) -> str:
    local = resolve_local_judge_config(judge_config)
    if local is not None:
        try:
            payload = _load_config_payload(local)
        except Exception:
            payload = None
        if payload is not None:
            resolved = _extract_model_like_value(payload)
            if resolved:
                return resolved

    resolved = _resolve_judge_model_from_alpaca_eval(judge_config)
    if resolved:
        return resolved

    inferred = _infer_judge_model_from_name(judge_config)
    if inferred:
        return inferred

    return judge_config
