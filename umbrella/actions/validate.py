"""Umbrella 🌂 — Validate Action"""

from ..schemas import ValidateResult, Backend
from ..backends.turboquant_pytorch    import PyTorchBackend
from ..backends.llamacpp_experimental import LlamaCppExperimentalBackend


def run_validate(
    bits:       int  = 4,
    seq_len:    int  = 512,
    synthetic:  bool = False,
    backend:    str  = Backend.PYTORCH,
    **kwargs,
) -> ValidateResult:
    if backend == Backend.LLAMACPP:
        return LlamaCppExperimentalBackend().validate(bits, seq_len, **kwargs)

    pt = PyTorchBackend()
    if synthetic or not pt.full_available:
        return pt._validate_synthetic(bits, seq_len)
    return pt._validate_full(bits, seq_len, **kwargs)
