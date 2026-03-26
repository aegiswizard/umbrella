"""
Umbrella 🌂 — Doctor Action
System capability detection. Works on every machine including CPU/Pi.
"""

import platform
import sys
from ..schemas import DoctorResult, Backend


def run_doctor() -> DoctorResult:
    r = DoctorResult()
    r.python_version = sys.version.split()[0]

    # RAM
    try:
        import psutil
        r.ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        r.ram_gb = 0.0

    # PyTorch
    try:
        import torch
        r.pytorch_available = True
        r.pytorch_version   = torch.__version__
        r.cuda_available    = torch.cuda.is_available()
        if r.cuda_available:
            r.cuda_version      = torch.version.cuda or ""
            r.cuda_device_count = torch.cuda.device_count()
            r.cuda_device_name  = torch.cuda.get_device_name(0)
            props               = torch.cuda.get_device_properties(0)
            r.vram_gb           = round(props.total_memory / (1024**3), 2)
    except ImportError:
        r.pytorch_available = False

    # TurboQuant core
    try:
        import turboquant  # noqa
        r.turboquant_core = True
    except ImportError:
        r.turboquant_core = False

    # llama.cpp
    import shutil
    r.llamacpp_available = bool(
        shutil.which("llama-cli") or shutil.which("llama-quantize")
    )

    # Capabilities
    r.full_validation_possible      = r.pytorch_available and r.cuda_available and r.turboquant_core
    r.synthetic_validation_possible = True  # Always

    if r.full_validation_possible:
        r.recommended_backend = Backend.PYTORCH
        r.capabilities.append("Full TurboQuant validation and compression (CUDA)")
        r.capabilities.append("Synthetic simulation")
        r.capabilities.append("suggest, doctor, validate, compress, serve — ALL modes")
    elif r.pytorch_available and r.cuda_available:
        r.recommended_backend = Backend.PYTORCH
        r.capabilities.append("Synthetic validation (turboquant not installed)")
        r.capabilities.append("suggest, doctor, validate --synthetic, serve")
        r.limitations.append("Install turboquant for full validation: pip install turboquant")
    elif r.pytorch_available:
        r.recommended_backend = Backend.SYNTHETIC
        r.capabilities.append("Synthetic validation (no CUDA)")
        r.capabilities.append("suggest, doctor, validate --synthetic, serve")
        r.limitations.append("CUDA GPU required for full validation and compression")
    else:
        r.recommended_backend = Backend.SYNTHETIC
        r.capabilities.append("Synthetic simulation — models expected behaviour")
        r.capabilities.append("suggest, doctor, serve")
        r.limitations.append("Install PyTorch for more: pip install torch")
        r.limitations.append("CUDA GPU required for full validation and compression")

    if r.llamacpp_available:
        r.capabilities.append("⚠️  EXPERIMENTAL: llama.cpp TurboQuant branch detected")

    # Verdict
    if r.full_validation_possible:
        r.verdict = "READY — Full TurboQuant validation and compression available"
    elif r.synthetic_validation_possible:
        r.verdict = "PARTIAL — Synthetic simulation available. Install turboquant + CUDA for full capability."
    else:
        r.verdict = "LIMITED — Basic suggest and planning only on this machine."

    return r
