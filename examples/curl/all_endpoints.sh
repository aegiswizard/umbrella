#!/usr/bin/env bash
# Umbrella 🌂 — curl examples
# Start server first: umbrella serve --port 8080 --mcp

BASE="http://localhost:8080"

echo "=== Health check ==="
curl -s "$BASE/health" | python3 -m json.tool

echo ""
echo "=== Doctor ==="
curl -s -X POST "$BASE/doctor" | python3 -m json.tool

echo ""
echo "=== Suggest: llama3-8b, 12GB VRAM, 32k context ==="
curl -s -X POST "$BASE/suggest" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3-8b","vram_gb":12,"context_length":32000,"quality_mode":"balanced"}' \
  | python3 -m json.tool

echo ""
echo "=== Suggest: aggressive quality ==="
curl -s -X POST "$BASE/suggest" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3-70b","vram_gb":40,"context_length":128000,"quality_mode":"aggressive"}' \
  | python3 -m json.tool

echo ""
echo "=== Validate: 4-bit synthetic ==="
curl -s -X POST "$BASE/validate" \
  -H "Content-Type: application/json" \
  -d '{"bits":4,"synthetic":true}' \
  | python3 -m json.tool

echo ""
echo "=== Autotune ==="
curl -s -X POST "$BASE/autotune" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3-8b","vram_gb":12,"context_length":32000,"quality_mode":"balanced"}' \
  | python3 -m json.tool

echo ""
echo "=== MCP manifest ==="
curl -s "$BASE/mcp" | python3 -m json.tool

echo ""
echo "=== MCP tool call: umbrella_autotune ==="
curl -s -X POST "$BASE/mcp" \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "umbrella_autotune",
    "parameters": {
      "model": "llama3-8b",
      "vram_gb": 12,
      "context_length": 32000,
      "quality_mode": "balanced"
    }
  }' | python3 -m json.tool
