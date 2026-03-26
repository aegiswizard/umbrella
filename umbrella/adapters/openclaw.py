"""
Umbrella 🌂 — OpenClaw Adapter
Registers Umbrella as an OpenClaw tool set.
Drop this file into your OpenClaw tools directory.
"""

from .python_agent import umbrella


def register_tools(registry):
    """Register all Umbrella tools with an OpenClaw tool registry."""

    @registry.tool(
        name="umbrella_doctor",
        description="Check what TurboQuant capabilities are available on this machine. Run this first.",
    )
    def doctor_tool(**_):
        return umbrella("doctor")

    @registry.tool(
        name="umbrella_suggest",
        description="Get optimal TurboQuant bit-width for your model and hardware.",
        parameters={
            "model":          {"type": "string",  "description": "Model name e.g. llama3-8b"},
            "vram_gb":        {"type": "number",  "description": "VRAM in GB"},
            "context_length": {"type": "integer", "description": "Context window in tokens"},
            "quality_mode":   {"type": "string",  "description": "safe | balanced | aggressive"},
        },
    )
    def suggest_tool(model="llama3-8b", vram_gb=0, context_length=8192, quality_mode="balanced", **_):
        return umbrella("suggest", model=model, vram_gb=vram_gb,
                        context_length=context_length, quality_mode=quality_mode)

    @registry.tool(
        name="umbrella_validate",
        description="Validate TurboQuant compression quality and return proof metrics.",
        parameters={
            "bits":      {"type": "integer", "description": "Bit-width e.g. 4"},
            "synthetic": {"type": "boolean", "description": "Use synthetic mode (no GPU needed)"},
        },
    )
    def validate_tool(bits=4, synthetic=False, **_):
        return umbrella("validate", bits=bits, synthetic=synthetic)

    @registry.tool(
        name="umbrella_autotune",
        description="Full decision engine. Returns optimal settings + validation plan.",
        parameters={
            "model":          {"type": "string"},
            "vram_gb":        {"type": "number"},
            "context_length": {"type": "integer"},
            "quality_mode":   {"type": "string"},
        },
    )
    def autotune_tool(model="llama3-8b", vram_gb=0, context_length=8192, quality_mode="balanced", **_):
        return umbrella("autotune", model=model, vram_gb=vram_gb,
                        context_length=context_length, quality_mode=quality_mode)
