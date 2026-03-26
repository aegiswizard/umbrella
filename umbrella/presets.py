"""
Umbrella 🌂 — Presets
TurboQuant bit-width profiles, hardware presets, and compression tables.
All values sourced from published TurboQuant research and MIT reference implementation.

References:
  - Google TurboQuant paper (2024): https://arxiv.org/abs/2412.09282
  - tonbistudio/turboquant-pytorch (MIT): https://github.com/tonbistudio/turboquant-pytorch
"""

from .schemas import QualityMode


# ---------------------------------------------------------------------------
# Bit-width profiles
# Source: TurboQuant paper Table 1 + MIT reference impl validation results
# ---------------------------------------------------------------------------

BIT_PROFILES = {
    2: {
        "name":                    "Ultra Aggressive",
        "bits":                    2,
        "expected_kv_reduction":   0.875,    # 87.5% size reduction
        "compression_ratio":       8.0,      # 8x smaller
        "attention_fidelity":      0.961,    # ~96.1% quality
        "topk_preservation":       0.94,
        "quality_mode":            QualityMode.AGGRESSIVE,
        "description":             "Maximum compression. Some quality degradation. Experimental.",
        "recommended_for":         "Research only. Very long context where quality loss is acceptable.",
        "warning":                 "Significant attention quality degradation. Not recommended for production.",
    },
    3: {
        "name":                    "Aggressive",
        "bits":                    3,
        "expected_kv_reduction":   0.8125,   # 81.25% size reduction
        "compression_ratio":       5.33,     # ~5.3x smaller
        "attention_fidelity":      0.995,    # ~99.5% quality (MIT impl reported)
        "topk_preservation":       0.97,
        "quality_mode":            QualityMode.AGGRESSIVE,
        "description":             "High compression with good quality. MIT reference reports 99.5% fidelity.",
        "recommended_for":         "Long context inference where VRAM is the primary constraint.",
        "warning":                 "Test thoroughly on your specific model before production use.",
    },
    4: {
        "name":                    "Balanced",
        "bits":                    4,
        "expected_kv_reduction":   0.75,     # 75% size reduction
        "compression_ratio":       4.0,      # 4x smaller
        "attention_fidelity":      0.998,    # ~99.8% quality
        "topk_preservation":       0.99,
        "quality_mode":            QualityMode.BALANCED,
        "description":             "Best balance of compression and quality. Recommended default.",
        "recommended_for":         "Most production use cases. Strong compression with minimal quality loss.",
        "warning":                 None,
    },
    6: {
        "name":                    "Conservative",
        "bits":                    6,
        "expected_kv_reduction":   0.625,    # 62.5% size reduction
        "compression_ratio":       2.67,     # ~2.7x smaller
        "attention_fidelity":      0.9995,   # ~99.95% quality
        "topk_preservation":       0.999,
        "quality_mode":            QualityMode.SAFE,
        "description":             "Conservative compression with near-lossless quality.",
        "recommended_for":         "Quality-critical applications where any degradation is unacceptable.",
        "warning":                 None,
    },
    8: {
        "name":                    "Safe",
        "bits":                    8,
        "expected_kv_reduction":   0.5,      # 50% size reduction
        "compression_ratio":       2.0,      # 2x smaller
        "attention_fidelity":      0.9999,
        "topk_preservation":       0.9999,
        "quality_mode":            QualityMode.SAFE,
        "description":             "Minimal compression, maximum quality preservation.",
        "recommended_for":         "First-time users, debugging, or highest quality requirements.",
        "warning":                 None,
    },
}


# ---------------------------------------------------------------------------
# Quality mode → default bit selection
# ---------------------------------------------------------------------------

QUALITY_TO_BITS = {
    QualityMode.SAFE:       8,
    QualityMode.BALANCED:   4,
    QualityMode.AGGRESSIVE: 3,
}


# ---------------------------------------------------------------------------
# KV cache memory estimation
# ---------------------------------------------------------------------------

# Approximate bytes per token per layer per head for FP16 (baseline)
# Formula: 2 (K+V) * num_heads * head_dim * 2 (bytes for fp16) per token per layer
KV_BYTES_PER_TOKEN_PER_LAYER_FP16 = 2  # simplified baseline factor

# Common model KV configurations
MODEL_CONFIGS = {
    # Model family patterns → (num_layers, num_kv_heads, head_dim)
    "llama3-8b":    (32, 8,  128),
    "llama3-70b":   (80, 8,  128),
    "llama3.1-8b":  (32, 8,  128),
    "llama3.1-70b": (80, 8,  128),
    "llama2-7b":    (32, 32, 128),
    "llama2-13b":   (40, 40, 128),
    "llama2-70b":   (80, 8,  128),
    "mistral-7b":   (32, 8,  128),
    "mistral-8x7b": (32, 8,  128),
    "phi3-mini":    (32, 32,  96),
    "phi3-medium":  (40, 40, 128),
    "gemma-2b":     (18, 1,  256),
    "gemma-7b":     (28, 16, 256),
    "qwen2-7b":     (28, 4,  128),
    "qwen2-72b":    (80, 8,  128),
    "codellama-7b": (32, 32, 128),
    "default":      (32, 8,  128),  # Fallback for unknown models
}


def estimate_kv_memory_gb(
    model_name: str,
    context_length: int,
    bits: int = 16,
) -> float:
    """
    Estimate KV cache memory in GB for a given model, context length, and bit-width.

    Formula based on TurboQuant paper section 3.1:
    KV_memory = num_layers * 2 * num_kv_heads * head_dim * context_length * (bits/8)

    Args:
        model_name:     Model name (matched against MODEL_CONFIGS)
        context_length: Token context window size
        bits:           Bit-width (16 = FP16 baseline)

    Returns:
        Estimated KV cache size in GB.
    """
    # Find config
    config = MODEL_CONFIGS.get("default")
    model_lower = model_name.lower().replace(" ", "").replace("-", "").replace("_", "")
    for key, val in MODEL_CONFIGS.items():
        key_norm = key.lower().replace(" ", "").replace("-", "").replace("_", "")
        if key_norm in model_lower or model_lower in key_norm:
            config = val
            break

    num_layers, num_kv_heads, head_dim = config
    bytes_per_element = bits / 8.0
    total_bytes = (
        num_layers * 2 * num_kv_heads * head_dim * context_length * bytes_per_element
    )
    return total_bytes / (1024 ** 3)


def get_profile(bits: int) -> dict:
    """Get the bit profile dict for a given bit-width."""
    return BIT_PROFILES.get(bits, BIT_PROFILES[4])


def recommend_bits(
    vram_gb: float,
    context_length: int,
    model_name: str,
    quality_mode: str = QualityMode.BALANCED,
) -> tuple:
    """
    Recommend the best bit-width given hardware and quality constraints.

    Returns:
        (recommended_bits, fallback_bits, reasoning_list)
    """
    reasoning = []
    baseline_kv = estimate_kv_memory_gb(model_name, context_length, bits=16)
    # Leave headroom for model weights and activations (rough estimate)
    available_for_kv = vram_gb * 0.35 if vram_gb > 0 else float("inf")

    reasoning.append(
        f"Baseline FP16 KV cache for {model_name} at {context_length:,} tokens: "
        f"{baseline_kv:.2f} GB"
    )
    reasoning.append(
        f"Available VRAM budget for KV cache (35% of {vram_gb} GB): "
        f"{available_for_kv:.2f} GB"
    )

    # Start from quality mode default
    start_bits = QUALITY_TO_BITS.get(quality_mode, 4)
    candidate_bits = [start_bits] + [b for b in sorted(BIT_PROFILES.keys()) if b != start_bits]

    recommended = start_bits
    fallback = 8

    for bits in candidate_bits:
        compressed_kv = estimate_kv_memory_gb(model_name, context_length, bits=bits)
        if compressed_kv <= available_for_kv:
            recommended = bits
            reasoning.append(
                f"{bits}-bit compression gives {compressed_kv:.2f} GB KV cache — fits in budget."
            )
            break
        else:
            reasoning.append(
                f"{bits}-bit gives {compressed_kv:.2f} GB — still exceeds budget of {available_for_kv:.2f} GB."
            )

    # Fallback is always one step more conservative
    fallback_candidates = [b for b in sorted(BIT_PROFILES.keys()) if b > recommended]
    fallback = fallback_candidates[0] if fallback_candidates else 16

    return recommended, fallback, reasoning


# ---------------------------------------------------------------------------
# Deployment presets
# ---------------------------------------------------------------------------

DEPLOYMENT_PRESETS = {
    2: {
        "preset_name":   "turbo-ultra",
        "use_case":      "Research / extreme long-context",
        "env_vars":      {"UMBRELLA_BITS": "2", "UMBRELLA_BACKEND": "pytorch"},
        "llama_cpp_arg": "--kv-bits 2",
        "warning":       "Experimental. Validate thoroughly before use.",
    },
    3: {
        "preset_name":   "turbo-aggressive",
        "use_case":      "Long-context inference, VRAM-constrained",
        "env_vars":      {"UMBRELLA_BITS": "3", "UMBRELLA_BACKEND": "pytorch"},
        "llama_cpp_arg": "--kv-bits 3",
        "warning":       None,
    },
    4: {
        "preset_name":   "turbo-balanced",
        "use_case":      "General purpose, recommended default",
        "env_vars":      {"UMBRELLA_BITS": "4", "UMBRELLA_BACKEND": "pytorch"},
        "llama_cpp_arg": "--kv-bits 4",
        "warning":       None,
    },
    6: {
        "preset_name":   "turbo-conservative",
        "use_case":      "Quality-critical applications",
        "env_vars":      {"UMBRELLA_BITS": "6", "UMBRELLA_BACKEND": "pytorch"},
        "llama_cpp_arg": "--kv-bits 6",
        "warning":       None,
    },
    8: {
        "preset_name":   "turbo-safe",
        "use_case":      "Testing, debugging, maximum quality",
        "env_vars":      {"UMBRELLA_BITS": "8", "UMBRELLA_BACKEND": "pytorch"},
        "llama_cpp_arg": "--kv-bits 8",
        "warning":       None,
    },
}
