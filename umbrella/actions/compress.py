"""Umbrella 🌂 — Compress Action"""

from ..schemas import CompressResult, Backend
from ..backends.turboquant_pytorch    import PyTorchBackend
from ..backends.llamacpp_experimental import LlamaCppExperimentalBackend


def run_compress(
    bits:        int = 4,
    backend:     str = Backend.PYTORCH,
    model_path:  str = "",
    output_path: str = "",
    **kwargs,
) -> CompressResult:
    if backend == Backend.LLAMACPP:
        return LlamaCppExperimentalBackend().compress(
            bits, model_path=model_path, output_path=output_path, **kwargs
        )
    return PyTorchBackend().compress(bits, model_path=model_path, output_path=output_path, **kwargs)
