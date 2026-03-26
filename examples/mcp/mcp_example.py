"""
Umbrella 🌂 — MCP usage example
Shows how to call Umbrella via MCP from any MCP-compatible agent (Claude, etc.)

1. Start the server:
   umbrella serve --port 8080 --mcp

2. Configure your agent to use: http://localhost:8080/mcp

3. The agent can then call these tools:
   - umbrella_doctor
   - umbrella_suggest
   - umbrella_validate
   - umbrella_autotune
"""

import urllib.request
import json


BASE = "http://localhost:8080"


def mcp_call(tool: str, parameters: dict = None) -> dict:
    payload = json.dumps({
        "tool":       tool,
        "parameters": parameters or {},
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/mcp",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def main():
    print("\n🌂 Umbrella MCP Example\n")

    # 1. Get MCP manifest
    print("── MCP Manifest ──────────────────────────────────────")
    req = urllib.request.Request(f"{BASE}/mcp")
    with urllib.request.urlopen(req, timeout=10) as resp:
        manifest = json.loads(resp.read())
    print(f"Name:  {manifest['name_for_human']}")
    print(f"Tools: {[t['name'] for t in manifest['tools']]}")

    # 2. Doctor via MCP
    print("\n── Doctor via MCP ────────────────────────────────────")
    result = mcp_call("umbrella_doctor")
    print(f"Verdict: {result['verdict']}")
    print(f"Backend: {result['recommended_backend']}")

    # 3. Suggest via MCP
    print("\n── Suggest via MCP ───────────────────────────────────")
    result = mcp_call("umbrella_suggest", {
        "model":          "llama3-8b",
        "vram_gb":        12,
        "context_length": 32000,
        "quality_mode":   "balanced",
    })
    print(f"Recommended bits:  {result['recommended_bits']}")
    print(f"Compression ratio: {result['expected_compression_ratio']}x")
    print(f"Memory saved:      {result['memory_saved_gb']} GB")

    # 4. Autotune via MCP
    print("\n── Autotune via MCP ──────────────────────────────────")
    result = mcp_call("umbrella_autotune", {
        "model":          "llama3-8b",
        "vram_gb":        12,
        "context_length": 32000,
        "quality_mode":   "balanced",
    })
    print(f"Chosen bits:    {result['chosen_bits']}")
    print(f"Chosen backend: {result['chosen_backend']}")
    print(f"Savings:        {result['expected_savings_gb']:.3f} GB")
    print("\nValidation plan:")
    for step in result.get("validation_plan", []):
        print(f"  · {step}")


if __name__ == "__main__":
    main()
