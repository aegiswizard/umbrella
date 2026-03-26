"""
Umbrella 🌂 — REST API
FastAPI application exposing all modes as HTTP endpoints.
Every endpoint returns structured JSON — machine-readable for agents.

Endpoints:
  GET  /health          Health check
  POST /doctor          System capability detection
  POST /suggest         Recommendation engine
  POST /validate        Compression validation
  POST /compress        Compression (CUDA required for full path)
  POST /autotune        Full decision engine
  GET  /mcp             MCP tool manifest (when --mcp enabled)
  POST /mcp             MCP tool calls
"""

from __future__ import annotations
import json
from typing import Optional


def create_app(enable_mcp: bool = False):
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
    except ImportError:
        raise ImportError(
            "FastAPI required for serve mode. Install with:\n"
            "  pip install 'umbrella-tq[serve]'\n"
            "  or: pip install fastapi uvicorn"
        )

    from umbrella.actions.doctor   import run_doctor
    from umbrella.actions.suggest  import run_suggest
    from umbrella.actions.validate import run_validate
    from umbrella.actions.compress import run_compress
    from umbrella.actions.autotune import run_autotune

    app = FastAPI(
        title="Umbrella 🌂 — Agent-Native TurboQuant",
        description=(
            "Agent-native KV cache compression and validation. "
            "CLI, HTTP, and MCP interfaces. "
            "MIT License — github.com/aegiswizard/umbrella"
        ),
        version="1.0.0",
    )

    # ── Request models ────────────────────────────────────────────────────

    class SuggestRequest(BaseModel):
        model:          str   = "llama3-8b"
        vram_gb:        float = 0.0
        context_length: int   = 8192
        quality_mode:   str   = "balanced"

    class ValidateRequest(BaseModel):
        bits:      int  = 4
        seq_len:   int  = 512
        synthetic: bool = False
        backend:   str  = "pytorch"

    class CompressRequest(BaseModel):
        bits:        int = 4
        backend:     str = "pytorch"
        model_path:  str = ""
        output_path: str = ""

    class AutotuneRequest(BaseModel):
        model:          str   = "llama3-8b"
        vram_gb:        float = 0.0
        context_length: int   = 8192
        quality_mode:   str   = "balanced"

    # ── Endpoints ─────────────────────────────────────────────────────────

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "umbrella", "version": "1.0.0"}

    @app.post("/doctor")
    def doctor():
        return JSONResponse(run_doctor().to_dict())

    @app.post("/suggest")
    def suggest(req: SuggestRequest):
        return JSONResponse(run_suggest(
            model_name=req.model,
            vram_gb=req.vram_gb,
            context_length=req.context_length,
            quality_mode=req.quality_mode,
        ).to_dict())

    @app.post("/validate")
    def validate(req: ValidateRequest):
        import dataclasses
        result = run_validate(
            bits=req.bits,
            seq_len=req.seq_len,
            synthetic=req.synthetic,
            backend=req.backend,
        )
        return JSONResponse(result.to_dict())

    @app.post("/compress")
    def compress(req: CompressRequest):
        result = run_compress(
            bits=req.bits,
            backend=req.backend,
            model_path=req.model_path,
            output_path=req.output_path,
        )
        return JSONResponse(result.to_dict())

    @app.post("/autotune")
    def autotune(req: AutotuneRequest):
        result = run_autotune(
            model_name=req.model,
            vram_gb=req.vram_gb,
            context_length=req.context_length,
            quality_mode=req.quality_mode,
        )
        return JSONResponse(result.to_dict())

    # ── MCP endpoints ─────────────────────────────────────────────────────

    if enable_mcp:
        MCP_TOOLS = [
            {
                "name":        "umbrella_doctor",
                "description": "Check what TurboQuant capabilities are available on this machine.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name":        "umbrella_suggest",
                "description": "Get optimal TurboQuant bit-width and settings for your model and hardware.",
                "inputSchema": {
                    "type":     "object",
                    "required": ["model", "vram_gb", "context_length"],
                    "properties": {
                        "model":          {"type": "string",  "description": "Model name e.g. llama3-8b"},
                        "vram_gb":        {"type": "number",  "description": "VRAM budget in GB (0 = unlimited)"},
                        "context_length": {"type": "integer", "description": "Context window in tokens"},
                        "quality_mode":   {"type": "string",  "enum": ["safe", "balanced", "aggressive"],
                                          "default": "balanced"},
                    },
                },
            },
            {
                "name":        "umbrella_validate",
                "description": "Validate TurboQuant compression quality. Returns proof metrics.",
                "inputSchema": {
                    "type":       "object",
                    "properties": {
                        "bits":      {"type": "integer", "default": 4},
                        "seq_len":   {"type": "integer", "default": 512},
                        "synthetic": {"type": "boolean", "default": False},
                    },
                },
            },
            {
                "name":        "umbrella_autotune",
                "description": "Full decision engine. Returns optimal settings + validation plan + fallback plan.",
                "inputSchema": {
                    "type":     "object",
                    "required": ["model", "vram_gb", "context_length"],
                    "properties": {
                        "model":          {"type": "string"},
                        "vram_gb":        {"type": "number"},
                        "context_length": {"type": "integer"},
                        "quality_mode":   {"type": "string", "enum": ["safe", "balanced", "aggressive"],
                                          "default": "balanced"},
                    },
                },
            },
        ]

        @app.get("/mcp")
        def mcp_manifest():
            return {
                "schema_version": "v1",
                "name_for_human": "Umbrella 🌂",
                "name_for_model": "umbrella",
                "description_for_human": "Agent-native TurboQuant KV cache compression and validation.",
                "description_for_model": (
                    "Use Umbrella to check hardware capabilities, recommend TurboQuant settings, "
                    "validate compression quality, and get proof-backed compression plans. "
                    "Call umbrella_doctor first to understand what this machine can do."
                ),
                "tools": MCP_TOOLS,
            }

        @app.post("/mcp")
        def mcp_call(body: dict):
            tool  = body.get("tool", "")
            params = body.get("parameters", {}) or body.get("input", {})

            dispatch = {
                "umbrella_doctor":   lambda: run_doctor().to_dict(),
                "umbrella_suggest":  lambda: run_suggest(
                    model_name=params.get("model", "llama3-8b"),
                    vram_gb=float(params.get("vram_gb", 0)),
                    context_length=int(params.get("context_length", 8192)),
                    quality_mode=params.get("quality_mode", "balanced"),
                ).to_dict(),
                "umbrella_validate": lambda: run_validate(
                    bits=int(params.get("bits", 4)),
                    seq_len=int(params.get("seq_len", 512)),
                    synthetic=bool(params.get("synthetic", False)),
                ).to_dict(),
                "umbrella_autotune": lambda: run_autotune(
                    model_name=params.get("model", "llama3-8b"),
                    vram_gb=float(params.get("vram_gb", 0)),
                    context_length=int(params.get("context_length", 8192)),
                    quality_mode=params.get("quality_mode", "balanced"),
                ).to_dict(),
            }

            fn = dispatch.get(tool)
            if not fn:
                return JSONResponse({"error": f"Unknown tool: {tool}"}, status_code=400)
            return JSONResponse(fn())

    return app
