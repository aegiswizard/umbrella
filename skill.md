# 🌂 Umbrella — Agent-Native TurboQuant Skill

**Version:** 1.0.0  
**License:** MIT  
**Source:** https://github.com/aegiswizard/umbrella  
**Compatible with:** OpenClaw · Hermes · Claude agents · Any Python agent · MCP clients

---

## What This Skill Does

Umbrella gives any agent the ability to inspect hardware, recommend TurboQuant
KV-cache compression settings, validate compression quality with proof metrics,
and generate deployment plans — all from one tool.

**Don't just quantize. Prove it fits.**

---

## Trigger Phrases

Your agent should invoke Umbrella when the user says:

- `"check if turboquant works on this machine"`
- `"what bits should I use for llama3-8b with 12GB VRAM"`
- `"validate turboquant at 4-bit"`
- `"how much memory will I save with turboquant"`
- `"compress my kv cache"`
- `"autotune turboquant for my setup"`
- `"umbrella doctor / suggest / validate / autotune"`
- `"will [model] fit in [X] GB with turboquant"`
- `"prove that 4-bit kv compression works"`

---

## Setup

```bash
git clone https://github.com/aegiswizard/umbrella.git
cd umbrella
pip install -e .

# For full validation (optional — needs CUDA GPU):
pip install torch turboquant

# For HTTP + MCP server:
pip install fastapi uvicorn

# For RAM detection:
pip install psutil
```

---

## CLI Usage

```bash
# 1. Check what this machine can do
umbrella doctor

# 2. Get recommendation for your model + hardware
umbrella suggest --model llama3-8b --vram 12 --context 32000
umbrella suggest --model llama3-70b --vram 80 --context 128000 --quality aggressive

# 3. Validate compression quality
umbrella validate --bits 4
umbrella validate --bits 3 --synthetic          # No GPU needed
umbrella validate --bits 4 --seq-len 1024

# 4. Full decision + proof plan
umbrella autotune --model llama3-8b --vram 12 --context 32000

# 5. Compress (requires CUDA + turboquant)
umbrella compress --bits 4

# 6. Start HTTP + MCP server
umbrella serve --port 8080
umbrella serve --port 8080 --mcp

# JSON output (agent-parseable)
umbrella suggest --model llama3-8b --vram 12 --context 32000 --output json
umbrella doctor --output json
```

---

## Python API

```python
from umbrella.adapters.python_agent import umbrella

# Check machine capabilities
result = umbrella("doctor")
print(result["verdict"])
print(result["full_validation_possible"])

# Get recommendation
result = umbrella("suggest", model="llama3-8b", vram_gb=12, context_length=32000)
print(result["recommended_bits"])      # e.g. 4
print(result["memory_saved_gb"])       # e.g. 2.4
print(result["compression_ratio"])     # e.g. 4.0
print(result["fits_in_budget"])        # True/False
print(result["report"])                # Full text report

# Validate
result = umbrella("validate", bits=4, synthetic=True)
print(result["status"])                # PASS / WARN / FAIL
print(result["metrics"])               # Full metrics dict

# Full autotune
result = umbrella("autotune", model="llama3-8b", vram_gb=12, context_length=32000)
print(result["chosen_bits"])
print(result["validation_plan"])
print(result["fallback_plan"])
print(result["report"])
```

---

## HTTP API (umbrella serve)

```bash
# Start server
umbrella serve --port 8080

# Health check
curl http://localhost:8080/health

# Doctor
curl -X POST http://localhost:8080/doctor

# Suggest
curl -X POST http://localhost:8080/suggest \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3-8b","vram_gb":12,"context_length":32000,"quality_mode":"balanced"}'

# Validate
curl -X POST http://localhost:8080/validate \
  -H "Content-Type: application/json" \
  -d '{"bits":4,"synthetic":true}'

# Autotune
curl -X POST http://localhost:8080/autotune \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3-8b","vram_gb":12,"context_length":32000}'
```

---

## MCP Server (umbrella serve --mcp)

```bash
umbrella serve --port 8080 --mcp
```

MCP manifest: `GET http://localhost:8080/mcp`

Tools exposed via MCP:
- `umbrella_doctor`
- `umbrella_suggest`
- `umbrella_validate`
- `umbrella_autotune`

---

## Output Keys — suggest

| Key | Type | Description |
|-----|------|-------------|
| `recommended_bits` | int | Optimal bit-width |
| `compression_ratio` | float | e.g. 4.0 = 4x smaller |
| `baseline_kv_memory_gb` | float | FP16 KV cache size |
| `compressed_kv_memory_gb` | float | After compression |
| `memory_saved_gb` | float | GB freed |
| `fits_in_budget` | bool | Fits in VRAM? |
| `attention_fidelity_estimate` | float | 0.0–1.0 |
| `fallback_bits` | int | Conservative fallback |
| `reasoning` | list | Step-by-step logic |
| `next_steps` | list | Exact commands to run |

## Output Keys — validate

| Key | Type | Description |
|-----|------|-------------|
| `status` | str | PASS / WARN / FAIL / SKIPPED |
| `mode` | str | "synthetic" or "full" |
| `metrics.compression_ratio` | float | Actual ratio achieved |
| `metrics.attention_similarity` | float | 0.0–1.0 quality measure |
| `metrics.topk_preservation` | float | Top-k attention preserved |
| `verdict` | str | Plain English verdict |

---

## Modes at a Glance

| Mode | GPU needed | Works on Pi | What it does |
|------|-----------|-------------|--------------|
| `doctor` | ❌ | ✅ | Detect capabilities |
| `suggest` | ❌ | ✅ | Recommend settings |
| `validate --synthetic` | ❌ | ✅ | Model expected results |
| `validate` | ✅ | ❌ | Real proof metrics |
| `compress` | ✅ | ❌ | Run compression |
| `autotune` | ❌ | ✅ | Full decision plan |
| `serve` | ❌ | ✅ | HTTP + MCP server |

---

## Bit-Width Reference

| Bits | Compression | Attention Fidelity | Quality Mode |
|------|------------|-------------------|--------------|
| 8 | 2.0x | ~99.99% | Safe |
| 6 | 2.7x | ~99.95% | Safe |
| 4 | 4.0x | ~99.8% | Balanced ⭐ |
| 3 | 5.3x | ~99.5% | Aggressive |
| 2 | 8.0x | ~96.1% | Ultra (experimental) |

---

## Disclaimer

Synthetic validation uses published TurboQuant paper numbers.
Real validation requires PyTorch 2+ + CUDA GPU + turboquant installed.
llama.cpp backend is experimental — validate on your model before production use.
