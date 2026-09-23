"""Device selection: CUDA (Colab / NVIDIA), MPS (Apple Silicon MacBook), else CPU."""
from __future__ import annotations

import os

# Apple Silicon: let the few ops MPS does not implement fall back to the CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def pick_device(requested: str | None = None) -> str:
    import torch

    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
