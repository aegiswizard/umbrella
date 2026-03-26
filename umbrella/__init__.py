"""
Umbrella 🌂 — Agent-Native TurboQuant
MIT License | github.com/aegiswizard/umbrella

Don't just quantize. Prove it fits.

Quick start:
    from umbrella.adapters.python_agent import umbrella

    result = umbrella("doctor")
    result = umbrella("suggest", model="llama3-8b", vram_gb=12, context_length=32000)
    result = umbrella("validate", bits=4)
    result = umbrella("autotune", model="llama3-8b", vram_gb=12, context_length=32000)
"""

__version__ = "1.0.0"
__author__  = "Aegis Wizard"
__license__ = "MIT"
__url__     = "https://github.com/aegiswizard/umbrella"
