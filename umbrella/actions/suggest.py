"""
Umbrella 🌂 — Suggest Action
Recommendation engine. Works on every machine. Pure math, no GPU needed.
"""

from ..schemas import SuggestResult, QualityMode, Backend
from ..presets import (
    estimate_kv_memory_gb, recommend_bits, get_profile, QUALITY_TO_BITS
)


def run_suggest(
    model_name:     str,
    vram_gb:        float,
    context_length: int,
    quality_mode:   str = QualityMode.BALANCED,
) -> SuggestResult:
    r = SuggestResult(
        model_name=model_name,
        vram_gb=vram_gb,
        context_length=context_length,
        quality_mode=quality_mode,
    )

    bits, fallback_bits, reasoning = recommend_bits(
        vram_gb=vram_gb,
        context_length=context_length,
        model_name=model_name,
        quality_mode=quality_mode,
    )

    profile       = get_profile(bits)
    baseline_kv   = estimate_kv_memory_gb(model_name, context_length, bits=16)
    compressed_kv = estimate_kv_memory_gb(model_name, context_length, bits=bits)
    saved         = baseline_kv - compressed_kv
    fits          = compressed_kv <= (vram_gb * 0.35) if vram_gb > 0 else True

    r.recommended_bits            = bits
    r.expected_kv_reduction       = round(profile["expected_kv_reduction"], 4)
    r.expected_compression_ratio  = round(profile["compression_ratio"], 3)
    r.attention_fidelity_estimate = profile["attention_fidelity"]
    r.baseline_kv_memory_gb       = round(baseline_kv, 3)
    r.compressed_kv_memory_gb     = round(compressed_kv, 3)
    r.memory_saved_gb             = round(saved, 3)
    r.fits_in_budget              = fits
    r.fallback_bits               = fallback_bits
    r.recommended_backend         = Backend.PYTORCH
    r.validation_available        = False  # Set by caller if CUDA present
    r.reasoning                   = reasoning

    r.next_steps = [
        f"Run: umbrella validate --bits {bits} --model {model_name}",
        f"If validation passes, run: umbrella compress --bits {bits}",
        f"Fallback option: --bits {fallback_bits} for more conservative compression",
    ]

    if not fits:
        r.warnings.append(
            f"Even {bits}-bit compression may not fit in {vram_gb} GB VRAM for this context length. "
            f"Consider reducing context length or using a smaller model."
        )
    if profile.get("warning"):
        r.warnings.append(profile["warning"])

    return r
