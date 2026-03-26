"""
Umbrella 🌂 — Python agent example
Shows every mode with real output.

Run: python examples/python/basic_usage.py
"""

from umbrella.adapters.python_agent import umbrella


def main():
    print("\n" + "="*56)
    print("  🌂 Umbrella — Python Agent Example")
    print("="*56 + "\n")

    # ── 1. Doctor ─────────────────────────────────────────────
    print("── 1. DOCTOR ──────────────────────────────────────────")
    result = umbrella("doctor")
    print(result["report"])
    print(f"Full validation possible: {result['full_validation_possible']}")
    print(f"Recommended backend:      {result['recommended_backend']}")

    # ── 2. Suggest ────────────────────────────────────────────
    print("── 2. SUGGEST ─────────────────────────────────────────")
    result = umbrella(
        "suggest",
        model="llama3-8b",
        vram_gb=12.0,
        context_length=32000,
        quality_mode="balanced",
    )
    print(result["report"])
    print(f"Recommended bits:  {result['recommended_bits']}")
    print(f"Compression ratio: {result['expected_compression_ratio']}x")
    print(f"Memory saved:      {result['memory_saved_gb']} GB")
    print(f"Fits in budget:    {result['fits_in_budget']}")

    # ── 3. Validate (synthetic — works everywhere) ────────────
    print("\n── 3. VALIDATE (synthetic) ────────────────────────────")
    result = umbrella("validate", bits=4, synthetic=True)
    print(result["report"])
    print(f"Status:  {result['status']}")
    if result.get("metrics"):
        m = result["metrics"]
        print(f"Ratio:   {m['compression_ratio']:.3f}x")
        print(f"Quality: {m['attention_similarity']*100:.2f}%")

    # ── 4. Autotune ───────────────────────────────────────────
    print("\n── 4. AUTOTUNE ────────────────────────────────────────")
    result = umbrella(
        "autotune",
        model="llama3-8b",
        vram_gb=12.0,
        context_length=32000,
        quality_mode="balanced",
    )
    print(result["report"])
    print(f"Chosen bits:     {result['chosen_bits']}")
    print(f"Chosen backend:  {result['chosen_backend']}")
    print(f"Expected savings:{result['expected_savings_gb']:.3f} GB")
    print("\nValidation plan:")
    for step in result.get("validation_plan", []):
        print(f"  · {step}")
    print("\nFallback plan:")
    for step in result.get("fallback_plan", []):
        print(f"  · {step}")

    # ── 5. JSON output (for programmatic agent use) ────────────
    print("\n── 5. JSON OUTPUT ─────────────────────────────────────")
    import json
    result = umbrella("suggest", model="llama3-8b", vram_gb=12.0, context_length=32000)
    print(json.dumps(result["data"], indent=2))


if __name__ == "__main__":
    main()
