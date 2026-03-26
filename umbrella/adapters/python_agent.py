"""
Umbrella 🌂 — Agent Adapter
One clean Python function for any agent to call.
OpenClaw · Hermes · Claude · Any Python agent.

Usage:
    from umbrella.agent import umbrella

    # Check machine
    result = umbrella("doctor")
    print(result["verdict"])

    # Get recommendation
    result = umbrella("suggest", model="llama3-8b", vram_gb=12, context_length=32000)
    print(result["recommended_bits"])
    print(result["memory_saved_gb"])

    # Full autotune
    result = umbrella("autotune", model="llama3-8b", vram_gb=12, context_length=32000)
    print(result["report"])
    print(result["chosen_bits"])
"""

from __future__ import annotations
from typing import Any

from umbrella.actions.doctor   import run_doctor
from umbrella.actions.suggest  import run_suggest
from umbrella.actions.validate import run_validate
from umbrella.actions.compress import run_compress
from umbrella.actions.autotune import run_autotune
from umbrella.report import (
    format_doctor, format_suggest, format_validate,
    format_compress, format_autotune,
)


def umbrella(mode: str, **kwargs) -> dict:
    """
    Universal entry point for all agent frameworks.

    Args:
        mode:   One of: "doctor" | "suggest" | "validate" | "compress" | "autotune"
        **kwargs: Mode-specific arguments (see below)

    Mode arguments:
        doctor:
            (no arguments)

        suggest:
            model          (str)   Model name, e.g. "llama3-8b"
            vram_gb        (float) VRAM in GB, 0 = unlimited/CPU
            context_length (int)   Context window in tokens
            quality_mode   (str)   "safe" | "balanced" | "aggressive"

        validate:
            bits       (int)   Bit-width, e.g. 4
            seq_len    (int)   Sequence length for test, default 512
            synthetic  (bool)  Force synthetic mode, default False
            backend    (str)   "pytorch" | "llamacpp_experimental"

        compress:
            bits        (int)  Bit-width
            backend     (str)  "pytorch" | "llamacpp_experimental"
            model_path  (str)  Path to model file (for llama.cpp)
            output_path (str)  Output path

        autotune:
            model          (str)   Model name
            vram_gb        (float) VRAM in GB
            context_length (int)   Context window in tokens
            quality_mode   (str)   "safe" | "balanced" | "aggressive"

    Returns:
        dict with keys:
            report     (str)   Human-readable text report
            data       (dict)  Raw structured result
            mode       (str)   Mode that was run
            + all fields from the specific result schema

    Raises:
        ValueError: If mode is not recognized.
    """
    mode = mode.lower().strip()

    if mode == "doctor":
        result = run_doctor()
        return {
            "report": format_doctor(result),
            "data":   result.to_dict(),
            "mode":   "doctor",
            **result.to_dict(),
        }

    elif mode == "suggest":
        result = run_suggest(
            model_name     = kwargs.get("model", "llama3-8b"),
            vram_gb        = float(kwargs.get("vram_gb", 0.0)),
            context_length = int(kwargs.get("context_length", 8192)),
            quality_mode   = kwargs.get("quality_mode", "balanced"),
        )
        return {
            "report": format_suggest(result),
            "data":   result.to_dict(),
            "mode":   "suggest",
            **result.to_dict(),
        }

    elif mode == "validate":
        result = run_validate(
            bits      = int(kwargs.get("bits", 4)),
            seq_len   = int(kwargs.get("seq_len", 512)),
            synthetic = bool(kwargs.get("synthetic", False)),
            backend   = kwargs.get("backend", "pytorch"),
        )
        return {
            "report": format_validate(result),
            "data":   result.to_dict(),
            "mode":   "validate",
            **result.to_dict(),
        }

    elif mode == "compress":
        result = run_compress(
            bits        = int(kwargs.get("bits", 4)),
            backend     = kwargs.get("backend", "pytorch"),
            model_path  = kwargs.get("model_path", ""),
            output_path = kwargs.get("output_path", ""),
        )
        return {
            "report": format_compress(result),
            "data":   result.to_dict(),
            "mode":   "compress",
            **result.to_dict(),
        }

    elif mode == "autotune":
        result = run_autotune(
            model_name     = kwargs.get("model", "llama3-8b"),
            vram_gb        = float(kwargs.get("vram_gb", 0.0)),
            context_length = int(kwargs.get("context_length", 8192)),
            quality_mode   = kwargs.get("quality_mode", "balanced"),
        )
        return {
            "report": format_autotune(result),
            "data":   result.to_dict(),
            "mode":   "autotune",
            **result.to_dict(),
        }

    else:
        raise ValueError(
            f"Unknown mode: '{mode}'. "
            "Valid modes: doctor | suggest | validate | compress | autotune"
        )
