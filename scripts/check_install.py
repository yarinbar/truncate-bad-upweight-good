"""Smoke-check the installed environment."""
import importlib
import sys

MODULES = ["torch", "vllm", "deepspeed", "flash_attn", "mpmath", "alpaca_eval", "transformers", "trl", "qrpo"]

for name in MODULES:
    module = importlib.import_module(name)
    print(f"{name:14s} {getattr(module, '__version__', 'ok')}")

import torch  # noqa: E402

if not torch.cuda.is_available():
    print("WARNING: CUDA is not available; training and evaluation need 4 GPUs.", file=sys.stderr)
else:
    print(f"CUDA devices: {torch.cuda.device_count()}")
