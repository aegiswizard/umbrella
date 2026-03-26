"""
Umbrella 🌂 — Autotune
The decision brain. Given your model, hardware, and quality goals —
returns the optimal TurboQuant configuration with a proof-backed plan.
"""

from ..schemas import AutotuneResult, QualityMode
from ..actions.suggest  import run_suggest
from ..actions.doctor   import run_doctor
from ..presets          import estimate_kv_memory_gb


def run_autotune(
    model_name:     str,
    vram_gb:        float,
    context_length: int,
    quality_mode:   str = QualityMode.BALANCED,
) -> AutotuneResult:
    r = AutotuneResult(
        model_name=model_name,
        vram_gb=vram_gb,
        context_length=context_length,
        quality_mode=quality_mode,
    )

    # 1. Detect hardware
    doctor = run_doctor()

    # 2. Get suggestion
    suggestion = run_suggest(
        model_name=model_name,
        vram_gb=vram_gb,
        context_length=context_length,
        quality_mode=quality_mode,
    )
    r.suggest             = suggestion
    r.chosen_bits         = suggestion.recommended_bits
    r.chosen_backend      = doctor.recommended_backend
    r.expected_savings_gb = suggestion.memory_saved_gb
    r.compression_ratio   = suggestion.expected_compression_ratio
    r.fits_in_budget      = suggestion.fits_in_budget
    r.reasoning           = suggestion.reasoning
    r.warnings            = suggestion.warnings

    # 3. Validation plan
    if doctor.full_validation_possible:
        r.validation_plan = [
            f"Run: umbrella validate --bits {r.chosen_bits} --seq-len 512",
            "Check attention similarity >= 0.995 for PASS",
            f"If PASS: run umbrella compress --bits {r.chosen_bits}",
        ]
    else:
        r.validation_plan = [
            f"Run: umbrella validate --bits {r.chosen_bits} --synthetic",
            "Synthetic result models expected behaviour from published TurboQuant numbers",
            "Install turboquant + CUDA GPU for real validation",
        ]

    # 4. Fallback plan
    r.fallback_plan = [
        f"If {r.chosen_bits}-bit quality is insufficient: retry with --bits {suggestion.fallback_bits}",
        f"Fallback memory estimate: {estimate_kv_memory_gb(model_name, context_length, suggestion.fallback_bits):.2f} GB",
        f"Reduce context length to {context_length // 2:,} tokens as an alternative",
    ]

    return r
