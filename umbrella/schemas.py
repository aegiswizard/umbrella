"""
Umbrella 🌂 — Schemas
All structured data types used across CLI, API, MCP, and agent interfaces.
Every output from every mode is one of these schemas — machine-readable JSON always.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json


# ---------------------------------------------------------------------------
# Enums as string constants (no enum import needed — simpler for agents)
# ---------------------------------------------------------------------------

class QualityMode:
    SAFE       = "safe"        # Maximum quality preservation, less compression
    BALANCED   = "balanced"    # Good quality, meaningful compression
    AGGRESSIVE = "aggressive"  # Maximum compression, some quality trade-off


class Backend:
    PYTORCH      = "pytorch"
    LLAMACPP     = "llamacpp_experimental"
    SYNTHETIC    = "synthetic"
    UNAVAILABLE  = "unavailable"


class ValidationStatus:
    PASS    = "PASS"
    WARN    = "WARN"
    FAIL    = "FAIL"
    SKIPPED = "SKIPPED"


# ---------------------------------------------------------------------------
# Doctor schema
# ---------------------------------------------------------------------------

@dataclass
class DoctorResult:
    """System capability report. Works on every machine."""
    python_version:         str  = ""
    pytorch_available:      bool = False
    pytorch_version:        str  = ""
    cuda_available:         bool = False
    cuda_version:           str  = ""
    cuda_device_count:      int  = 0
    cuda_device_name:       str  = ""
    vram_gb:                float = 0.0
    ram_gb:                 float = 0.0
    turboquant_core:        bool = False
    llamacpp_available:     bool = False
    full_validation_possible: bool = False
    synthetic_validation_possible: bool = False
    recommended_backend:    str  = Backend.UNAVAILABLE
    capabilities:           list = field(default_factory=list)
    limitations:            list = field(default_factory=list)
    verdict:                str  = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Suggest schema
# ---------------------------------------------------------------------------

@dataclass
class SuggestResult:
    """Recommendation for optimal TurboQuant settings. Works on every machine."""
    model_name:             str   = ""
    vram_gb:                float = 0.0
    context_length:         int   = 0
    quality_mode:           str   = QualityMode.BALANCED
    recommended_bits:       int   = 4
    expected_kv_reduction:  float = 0.0   # e.g. 0.75 = 75% reduction
    expected_compression_ratio: float = 0.0  # e.g. 4.0 = 4x smaller
    attention_fidelity_estimate: float = 0.0  # 0.0–1.0
    baseline_kv_memory_gb:  float = 0.0
    compressed_kv_memory_gb: float = 0.0
    memory_saved_gb:        float = 0.0
    fits_in_budget:         bool  = False
    recommended_backend:    str   = Backend.SYNTHETIC
    validation_available:   bool  = False
    fallback_bits:          int   = 8
    reasoning:              list  = field(default_factory=list)
    warnings:               list  = field(default_factory=list)
    next_steps:             list  = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Validate schema
# ---------------------------------------------------------------------------

@dataclass
class ValidationMetrics:
    """Detailed metrics from a validation run."""
    baseline_memory_mb:    float = 0.0
    compressed_memory_mb:  float = 0.0
    compression_ratio:     float = 0.0
    attention_similarity:  float = 0.0   # 0.0–1.0 cosine similarity
    topk_preservation:     float = 0.0   # 0.0–1.0 fraction of top-k preserved
    bits_used:             int   = 4
    sequence_length:       int   = 0
    head_dim:              int   = 0
    num_heads:             int   = 0


@dataclass
class ValidateResult:
    """Proof-backed validation result."""
    status:           str   = ValidationStatus.SKIPPED
    mode:             str   = "synthetic"   # "synthetic" | "full"
    backend:          str   = Backend.SYNTHETIC
    bits:             int   = 4
    metrics:          Optional[ValidationMetrics] = None
    pass_threshold:   float = 0.95
    warn_threshold:   float = 0.90
    verdict:          str   = ""
    detail:           str   = ""
    warnings:         list  = field(default_factory=list)
    hardware_used:    str   = ""

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Compress schema
# ---------------------------------------------------------------------------

@dataclass
class CompressResult:
    """Result of a compression operation."""
    status:              str   = "unavailable"
    backend:             str   = Backend.UNAVAILABLE
    bits:                int   = 4
    input_description:   str   = ""
    output_path:         str   = ""
    metadata_path:       str   = ""
    compression_ratio:   float = 0.0
    memory_before_mb:    float = 0.0
    memory_after_mb:     float = 0.0
    quality_verified:    bool  = False
    validation_result:   Optional[ValidateResult] = None
    deployment_preset:   dict  = field(default_factory=dict)
    warnings:            list  = field(default_factory=list)
    detail:              str   = ""

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Autotune schema
# ---------------------------------------------------------------------------

@dataclass
class AutotuneResult:
    """Full decision from the autotune brain."""
    model_name:          str   = ""
    vram_gb:             float = 0.0
    context_length:      int   = 0
    quality_mode:        str   = QualityMode.BALANCED
    chosen_bits:         int   = 4
    chosen_backend:      str   = Backend.SYNTHETIC
    expected_savings_gb: float = 0.0
    compression_ratio:   float = 0.0
    fits_in_budget:      bool  = False
    validation_plan:     list  = field(default_factory=list)
    fallback_plan:       list  = field(default_factory=list)
    reasoning:           list  = field(default_factory=list)
    warnings:            list  = field(default_factory=list)
    suggest:             Optional[SuggestResult] = None

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)

    def to_dict(self) -> dict:
        return asdict(self)
