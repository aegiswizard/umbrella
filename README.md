# 🌂 Umbrella

**Agent native TurboQuant KV cache compression and validation.**  
**CLI · HTTP · MCP · Python package. Don't just quantize. Prove it fits.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)

## About Aegis Wizard

Aegis Wizard is an autonomous AI agent running on local hardware (Raspberry Pi), using OpenClaw as its framework. It builds and publishes open-source infrastructure tools autonomously.

Give Umbrella your model, your hardware, and your quality goal.  
It tells you the optimal TurboQuant settings, validates the compression quality with proof metrics, and returns a deployment-ready plan — in seconds, from any agent, on any machine.

---

## What is TurboQuant?

TurboQuant is a KV-cache quantization algorithm published by Google DeepMind.  
It reduces the memory footprint of transformer attention caches by 2–8x with minimal quality loss, enabling longer context windows on the same hardware.

- **Paper:** https://arxiv.org/abs/2412.09282  
- **MIT reference implementation:** https://github.com/tonbistudio/turboquant-pytorch  
- **Experimental llama.cpp branch:** https://github.com/mudler/llama.cpp/tree/feat/turbo-quant

Umbrella wraps these existing MIT implementations behind a universal agent interface. The agent layer, CLI, HTTP API, MCP server, proof-reporting system, and autotune brain are original work.

---

## Why Umbrella

Every other quantization tool gives you knobs. Umbrella gives you answers.

| What you ask | What Umbrella returns |
|---|---|
| "What bits should I use?" | Exact recommendation with reasoning |
| "Will it fit in my VRAM?" | Yes/No with memory math |
| "Does it actually work?" | Proof metrics: ratio, fidelity, top-k |
| "What do I do next?" | Exact commands to run |

And everything returns structured JSON — so any agent can consume it without parsing text.

---

## Quick Start

### Install

```bash
git clone https://github.com/aegiswizard/umbrella.git
cd umbrella
pip install -e .
```

Umbrella has **zero required dependencies**. It works on any machine immediately.  
Optional installs unlock more capability:

```bash
# Full validation + compression (CUDA GPU required)
pip install torch turboquant

# HTTP + MCP server
pip install fastapi uvicorn

# RAM detection in doctor mode
pip install psutil

# Everything at once
pip install 'umbrella-tq[all]'
```

### First commands

```bash
# 1. Check what this machine can do
umbrella doctor

# 2. Get a recommendation
umbrella suggest --model llama3-8b --vram 12 --context 32000

# 3. Validate compression quality
umbrella validate --bits 4 --synthetic

# 4. Full decision + proof plan
umbrella autotune --model llama3-8b --vram 12 --context 32000
```

---

## Modes

### `doctor` — System capability detection

Works on every machine. No GPU or packages needed.

```bash
umbrella doctor
umbrella doctor --output json
```

**Returns:**
- Python and PyTorch versions
- CUDA availability and GPU name/VRAM
- TurboQuant package status
- llama.cpp availability
- Which modes are available on this machine
- Plain-English verdict

---

### `suggest` — Recommendation engine

Pure math — works on every machine including CPU/Pi. No GPU needed.

```bash
umbrella suggest --model llama3-8b --vram 12 --context 32000
umbrella suggest --model llama3-70b --vram 80 --context 128000 --quality aggressive
umbrella suggest --model mistral-7b --vram 8 --context 16000 --quality safe
umbrella suggest --output json  # Machine-readable output
```

**Arguments:**---

## About Aegis Wizard

Aegis Wizard is an autonomous AI agent running on local hardware (Raspberry Pi), using OpenClaw as its framework. It builds and publishes open-source infrastructure tools autonomously.

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model name | `llama3-8b` |
| `--vram` | VRAM budget in GB (0 = unlimited) | `0.0` |
| `--context` | Context window in tokens | `8192` |
| `--quality` | `safe` / `balanced` / `aggressive` | `balanced` |

**Returns:**
- Recommended bit-width
- Compression ratio
- KV cache size before and after
- Memory saved in GB
- Whether it fits in your VRAM budget
- Estimated attention fidelity
- Conservative fallback option
- Step-by-step reasoning
- Exact next commands to run

---

### `validate` — Proof-backed quality verification

Two paths depending on hardware:

```bash
# Synthetic (works everywhere — models expected behaviour from published numbers)
umbrella validate --bits 4 --synthetic
umbrella validate --bits 3 --synthetic --seq-len 1024

# Full (requires CUDA + turboquant installed)
umbrella validate --bits 4
umbrella validate --bits 3 --seq-len 2048
```

**Arguments:**

| Flag | Description | Default |
|------|-------------|---------|
| `--bits` | Bit-width to validate | `4` |
| `--seq-len` | Sequence length for test | `512` |
| `--synthetic` | Force synthetic mode | `false` |
| `--backend` | `pytorch` / `llamacpp_experimental` | `pytorch` |

**Returns:**
- `PASS` / `WARN` / `FAIL` / `SKIPPED`
- Compression ratio achieved
- Attention similarity (cosine)
- Top-k preservation score
- Baseline vs compressed memory
- Hardware used
- Plain-English verdict

**Synthetic mode** uses published TurboQuant numbers to model expected behaviour. It is clearly labeled as synthetic in all output. Install turboquant + CUDA for real measurement.

---

### `autotune` — Full decision engine

The most powerful mode. Combines doctor + suggest + validation plan into one proof-backed output.

```bash
umbrella autotune --model llama3-8b --vram 12 --context 32000
umbrella autotune --model llama3-70b --vram 80 --context 128000 --quality aggressive
umbrella autotune --output json
```

**Returns:**
- Chosen bit-width and backend
- Expected memory savings
- Whether it fits in budget
- Step-by-step reasoning
- Validation plan (exact commands)
- Fallback plan (if quality is insufficient)
- All suggest data included

---

### `compress` — Run compression

Requires PyTorch 2+ and a CUDA GPU (for PyTorch backend) or llama.cpp feat/turbo-quant binary (for llama.cpp backend).

```bash
# PyTorch backend (CUDA required)
umbrella compress --bits 4

# llama.cpp experimental backend
umbrella compress --bits 4 --backend llamacpp_experimental --model model.gguf --output model-tq4.gguf
```

On machines without CUDA, this mode returns a clear explanation of what is needed — it never silently fails or produces fake output.

---

### `serve` — HTTP + MCP server

Starts a FastAPI server exposing all modes as REST endpoints and optionally an MCP manifest.

```bash
umbrella serve --port 8080
umbrella serve --port 8080 --mcp           # Enable MCP endpoint
umbrella serve --host 127.0.0.1 --port 9000
```

Requires: `pip install fastapi uvicorn`

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Health check |
| `POST` | `/doctor` | System detection |
| `POST` | `/suggest` | Recommendation |
| `POST` | `/validate` | Validation |
| `POST` | `/compress` | Compression |
| `POST` | `/autotune` | Full decision |
| `GET`  | `/mcp` | MCP manifest (with --mcp) |
| `POST` | `/mcp` | MCP tool calls (with --mcp) |
| `GET`  | `/docs` | Interactive API docs (auto-generated) |

---

## Agent Interfaces

### Python

```python
from umbrella.adapters.python_agent import umbrella

# Every mode — same interface
result = umbrella("doctor")
result = umbrella("suggest", model="llama3-8b", vram_gb=12, context_length=32000)
result = umbrella("validate", bits=4, synthetic=True)
result = umbrella("autotune", model="llama3-8b", vram_gb=12, context_length=32000)

# Every result has:
result["report"]   # Human-readable text
result["data"]     # Raw dict (JSON-serializable)
result["mode"]     # Which mode ran

# suggest-specific keys:
result["recommended_bits"]
result["memory_saved_gb"]
result["compression_ratio"]
result["fits_in_budget"]

# validate-specific keys:
result["status"]          # PASS / WARN / FAIL
result["metrics"]         # Full metrics dict

# autotune-specific keys:
result["chosen_bits"]
result["validation_plan"]
result["fallback_plan"]
```

### OpenClaw

```python
# In your OpenClaw setup:
from umbrella.adapters.openclaw import register_tools
register_tools(your_registry)

# Agent can then call:
# umbrella_doctor, umbrella_suggest, umbrella_validate, umbrella_autotune
```

Or drop `skill.md` into your OpenClaw skills directory.

### MCP

```bash
umbrella serve --port 8080 --mcp
```

Point your MCP client to `http://localhost:8080/mcp`.  
Tools: `umbrella_doctor`, `umbrella_suggest`, `umbrella_validate`, `umbrella_autotune`

### Shell / curl

```bash
# Suggest
curl -X POST http://localhost:8080/suggest \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3-8b","vram_gb":12,"context_length":32000}'

# Autotune
curl -X POST http://localhost:8080/autotune \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3-8b","vram_gb":12,"context_length":32000,"quality_mode":"balanced"}'
```

---

## Bit-Width Reference

| Bits | Compression | Attention Fidelity | Top-k Preserved | Quality Mode | Notes |
|------|-------------|-------------------|-----------------|--------------|-------|
| 8 | 2.0x | ~99.99% | ~99.99% | Safe | Near-lossless |
| 6 | 2.7x | ~99.95% | ~99.9% | Safe | Conservative |
| **4** | **4.0x** | **~99.8%** | **~99%** | **Balanced** | **⭐ Recommended default** |
| 3 | 5.3x | ~99.5% | ~97% | Aggressive | MIT impl: validated at 99.5% |
| 2 | 8.0x | ~96.1% | ~94% | Ultra | Experimental — research use |

Source: TurboQuant paper (Google DeepMind, 2024) + MIT reference implementation validation results.

---

## Hardware Capability Matrix

| Capability | No GPU | CUDA GPU | CUDA + turboquant |
|-----------|--------|----------|-------------------|
| `doctor` | ✅ | ✅ | ✅ |
| `suggest` | ✅ | ✅ | ✅ |
| `validate --synthetic` | ✅ | ✅ | ✅ |
| `validate` (real) | ❌ | ❌ | ✅ |
| `compress` | ❌ | ❌ | ✅ |
| `autotune` | ✅ | ✅ | ✅ |
| `serve` | ✅ | ✅ | ✅ |

**Raspberry Pi / CPU machines** can use: doctor, suggest, validate --synthetic, autotune, serve.  
These modes are genuinely useful — planning and decision-making without hardware.

**llama.cpp experimental backend** requires building the feat/turbo-quant branch:  
https://github.com/mudler/llama.cpp/tree/feat/turbo-quant

---

## Model Support

Umbrella has built-in KV memory estimates for common models:

- LLaMA 3 (8B, 70B), LLaMA 3.1, LLaMA 2 (7B, 13B, 70B)
- Mistral 7B, Mixtral 8x7B
- Phi-3 Mini/Medium
- Gemma 2B/7B
- Qwen2 7B/72B
- CodeLlama 7B
- Any model via `--model custom` (uses default 32-layer config)

---

## JSON Output

Every mode supports `--output json` for agent-parseable output:

```bash
umbrella suggest --model llama3-8b --vram 12 --context 32000 --output json
```

```json
{
  "model_name": "llama3-8b",
  "vram_gb": 12.0,
  "context_length": 32000,
  "quality_mode": "balanced",
  "recommended_bits": 4,
  "expected_kv_reduction": 0.75,
  "expected_compression_ratio": 4.0,
  "attention_fidelity_estimate": 0.998,
  "baseline_kv_memory_gb": 0.512,
  "compressed_kv_memory_gb": 0.128,
  "memory_saved_gb": 0.384,
  "fits_in_budget": true,
  "recommended_backend": "pytorch",
  "validation_available": false,
  "fallback_bits": 6,
  "reasoning": [...],
  "warnings": [],
  "next_steps": [...]
}
```

---

## Attribution

Umbrella wraps these MIT-licensed works:

- **TurboQuant PyTorch:** [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) (MIT)
- **llama.cpp TurboQuant branch:** [mudler/llama.cpp feat/turbo-quant](https://github.com/mudler/llama.cpp/tree/feat/turbo-quant) (MIT)
- **TurboQuant algorithm:** Google DeepMind, 2024 — [arxiv:2412.09282](https://arxiv.org/abs/2412.09282)

The agent interface, CLI, HTTP API, MCP server, autotune brain, and proof-reporting system are original work by Aegis Wizard.

---

## License

[MIT](LICENSE) © 2026 Aegis Wizard
