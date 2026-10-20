import re
from collections.abc import Iterable, Sequence
from pathlib import Path


CHECKPOINT_NAME_RE = re.compile(r"^checkpoint-(\d+)$")
POSITIVE_INTEGER_RE = re.compile(r"^[1-9]\d*$")


def normalize_checkpoint_steps(
    requested_steps: Iterable[object] | None,
) -> list[int]:
    """Validate, deduplicate, and sort requested checkpoint steps."""
    if requested_steps is None:
        return []

    normalized_steps: set[int] = set()
    for requested_step in requested_steps:
        if isinstance(requested_step, bool):
            raise ValueError(
                f"Checkpoint step must be a positive integer: {requested_step!r}"
            )

        step_text = str(requested_step).strip()
        if POSITIVE_INTEGER_RE.fullmatch(step_text) is None:
            raise ValueError(
                f"Checkpoint step must be a positive integer: {requested_step!r}"
            )
        normalized_steps.add(int(step_text))

    return sorted(normalized_steps)


def select_checkpoint_paths(
    checkpoints: Sequence[Path],
    requested_steps: Iterable[object] | None,
    *,
    context: Path,
) -> list[Path]:
    """Select requested checkpoint paths while preserving legacy no-filter order."""
    checkpoint_paths = list(checkpoints)
    normalized_steps = normalize_checkpoint_steps(requested_steps)
    if not normalized_steps:
        return checkpoint_paths

    checkpoints_by_step: dict[int, Path] = {}
    for checkpoint_path in checkpoint_paths:
        match = CHECKPOINT_NAME_RE.fullmatch(checkpoint_path.name)
        if match is not None:
            checkpoints_by_step[int(match.group(1))] = checkpoint_path

    missing_steps = [
        step for step in normalized_steps if step not in checkpoints_by_step
    ]
    if missing_steps:
        raise ValueError(
            f"Missing requested checkpoint steps {missing_steps} in {context}"
        )

    return [checkpoints_by_step[step] for step in normalized_steps]
