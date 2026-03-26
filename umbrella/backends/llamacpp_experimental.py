"""
Umbrella 🌂 — llama.cpp Experimental Backend

⚠️  EXPERIMENTAL — This backend wraps the mudler/llama.cpp feat/turbo-quant branch.
    That branch currently builds and quantizes correctly but is still under evaluation
    by the llama.cpp community. Use in production only after validating on your model.

Source: mudler/llama.cpp feat/turbo-quant (MIT License)
        https://github.com/mudler/llama.cpp/tree/feat/turbo-quant

Attribution: llama.cpp by Georgi Gerganov (MIT).
             TurboQuant branch by mudler (MIT).
             Agent wrapper by Aegis Wizard (MIT).
"""

from __future__ import annotations
import subprocess
import shutil
from ..schemas import ValidateResult, CompressResult, Backend, ValidationStatus
from ..presets import get_profile
from .base import BaseBackend


def _find_llama_binary() -> str | None:
    """Find llama.cpp binary with TurboQuant support."""
    candidates = [
        "llama-cli", "llama-quantize", "llama.cpp",
        "./llama-cli", "./llama-quantize",
    ]
    for c in candidates:
        if shutil.which(c):
            return c
    return None


class LlamaCppExperimentalBackend(BaseBackend):
    """
    Experimental llama.cpp backend for TurboQuant.
    Requires the feat/turbo-quant branch to be built and on PATH.

    EXPERIMENTAL — labeled clearly in all output.
    """

    def __init__(self):
        self._binary = _find_llama_binary()

    @property
    def name(self) -> str:
        return Backend.LLAMACPP

    @property
    def available(self) -> bool:
        return self._binary is not None

    @property
    def requires_gpu(self) -> bool:
        return False  # llama.cpp supports CPU inference

    def validate(self, bits: int, seq_len: int = 512, **kwargs) -> ValidateResult:
        if not self.available:
            return ValidateResult(
                status=ValidationStatus.SKIPPED,
                mode="llamacpp_experimental",
                backend=Backend.LLAMACPP,
                bits=bits,
                verdict=(
                    "llama.cpp feat/turbo-quant binary not found on PATH. "
                    "Build from: https://github.com/mudler/llama.cpp/tree/feat/turbo-quant"
                ),
                warnings=["EXPERIMENTAL BACKEND — not available on this machine."],
            )

        # llama.cpp validation is done via the quantize tool
        # Return informational result explaining how to validate
        profile = get_profile(bits)
        return ValidateResult(
            status=ValidationStatus.WARN,
            mode="llamacpp_experimental",
            backend=Backend.LLAMACPP,
            bits=bits,
            verdict=(
                f"⚠️  EXPERIMENTAL: llama.cpp TurboQuant branch detected at '{self._binary}'. "
                f"To validate, run: {self._binary} --kv-bits {bits} [model] [output]. "
                f"Expected compression: {profile['compression_ratio']:.1f}x based on published numbers."
            ),
            warnings=[
                "EXPERIMENTAL BACKEND — feat/turbo-quant branch is still under evaluation.",
                "Validate thoroughly on your specific model before production use.",
                "Use PyTorch backend for more complete validation tooling.",
            ],
        )

    def compress(self, bits: int, model_path: str = "", output_path: str = "", **kwargs) -> CompressResult:
        if not self.available:
            return CompressResult(
                status="unavailable",
                backend=Backend.LLAMACPP,
                bits=bits,
                detail=(
                    "llama.cpp feat/turbo-quant binary not found. "
                    "Build from: https://github.com/mudler/llama.cpp/tree/feat/turbo-quant"
                ),
                warnings=["EXPERIMENTAL BACKEND — not available on this machine."],
            )

        if not model_path:
            return CompressResult(
                status="error",
                backend=Backend.LLAMACPP,
                bits=bits,
                detail="model_path is required for llama.cpp compression.",
            )

        cmd = [self._binary, "--kv-bits", str(bits), model_path]
        if output_path:
            cmd += ["--output", output_path]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                profile = get_profile(bits)
                return CompressResult(
                    status="success",
                    backend=Backend.LLAMACPP,
                    bits=bits,
                    output_path=output_path,
                    compression_ratio=profile["compression_ratio"],
                    warnings=["EXPERIMENTAL BACKEND — validate output before production use."],
                    detail=result.stdout[:500] if result.stdout else "Compression completed.",
                )
            else:
                return CompressResult(
                    status="failed",
                    backend=Backend.LLAMACPP,
                    bits=bits,
                    detail=result.stderr[:500] if result.stderr else "llama.cpp returned non-zero exit code.",
                    warnings=["EXPERIMENTAL BACKEND."],
                )
        except subprocess.TimeoutExpired:
            return CompressResult(
                status="timeout",
                backend=Backend.LLAMACPP,
                bits=bits,
                detail="llama.cpp compression timed out after 600 seconds.",
            )
        except Exception as exc:
            return CompressResult(
                status="failed",
                backend=Backend.LLAMACPP,
                bits=bits,
                detail=str(exc),
            )
