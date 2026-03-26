"""
Umbrella 🌂 — CLI
umbrella doctor | suggest | validate | compress | autotune | serve
"""

import argparse
import sys
import json

BANNER = """
  🌂  Umbrella — Agent-Native TurboQuant  v1.0.0
      Don't just quantize. Prove it fits.
      github.com/aegiswizard/umbrella  ·  MIT License
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="umbrella",
        description="🌂 Umbrella — Agent-native KV cache compression and validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
commands:
  doctor                     Check what this machine can do
  suggest                    Get optimal TurboQuant settings for your hardware
  validate                   Validate compression quality (synthetic or real)
  compress                   Run compression (requires CUDA + turboquant)
  autotune                   Full decision engine — hardware + settings + proof plan
  serve                      Start HTTP + MCP API server

examples:
  umbrella doctor
  umbrella suggest --model llama3-8b --vram 12 --context 32000
  umbrella suggest --model llama3-70b --vram 40 --context 128000 --quality aggressive
  umbrella validate --bits 4
  umbrella validate --bits 3 --synthetic
  umbrella compress --bits 4
  umbrella autotune --model llama3-8b --vram 12 --context 32000
  umbrella serve --port 8080
  umbrella serve --port 8080 --mcp
        """,
    )
    parser.add_argument("--output", "-o", choices=["text", "json"], default="text")

    sub = parser.add_subparsers(dest="command", metavar="command")

    # doctor
    sub.add_parser("doctor", help="Detect machine capabilities")

    # suggest
    s = sub.add_parser("suggest", help="Recommend optimal bit-width and settings")
    s.add_argument("--model",   "-m", default="llama3-8b", help="Model name (default: llama3-8b)")
    s.add_argument("--vram",    "-v", type=float, default=0.0, help="VRAM in GB (0 = unlimited/CPU)")
    s.add_argument("--context", "-c", type=int, default=8192,  help="Context length in tokens")
    s.add_argument("--quality", "-q", default="balanced",
                   choices=["safe", "balanced", "aggressive"], help="Quality mode")

    # validate
    v = sub.add_parser("validate", help="Validate compression quality")
    v.add_argument("--bits",      "-b", type=int, default=4,   help="Bit-width to validate")
    v.add_argument("--seq-len",   "-s", type=int, default=512, help="Sequence length for test")
    v.add_argument("--synthetic", action="store_true",          help="Force synthetic mode")
    v.add_argument("--backend",   default="pytorch",
                   choices=["pytorch", "llamacpp_experimental"])

    # compress
    c = sub.add_parser("compress", help="Run compression (CUDA required for full path)")
    c.add_argument("--bits",    "-b", type=int, default=4,  help="Bit-width")
    c.add_argument("--model",   "-m", default="",           help="Path to model file (llama.cpp)")
    c.add_argument("--output",  "-O", default="",           dest="out_path", help="Output path")
    c.add_argument("--backend", default="pytorch",
                   choices=["pytorch", "llamacpp_experimental"])

    # autotune
    a = sub.add_parser("autotune", help="Full decision: hardware + settings + proof plan")
    a.add_argument("--model",   "-m", default="llama3-8b")
    a.add_argument("--vram",    "-v", type=float, default=0.0)
    a.add_argument("--context", "-c", type=int,   default=8192)
    a.add_argument("--quality", "-q", default="balanced",
                   choices=["safe", "balanced", "aggressive"])

    # serve
    sv = sub.add_parser("serve", help="Start HTTP + MCP API server")
    sv.add_argument("--port", "-p", type=int, default=8080)
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--mcp",  action="store_true", help="Enable MCP endpoint")

    # version
    sub.add_parser("version", help="Print version")

    args = parser.parse_args()

    if not args.command:
        print(BANNER)
        parser.print_help()
        sys.exit(0)

    if args.command == "version":
        print("umbrella 1.0.0")
        sys.exit(0)

    # Determine output format — prefer --output from before subcommand
    fmt = getattr(args, "output", "text")

    print(BANNER, file=sys.stderr)

    if args.command == "doctor":
        from umbrella.actions.doctor import run_doctor
        from umbrella.report import format_doctor
        result = run_doctor()
        print(format_doctor(result, fmt))

    elif args.command == "suggest":
        from umbrella.actions.suggest import run_suggest
        from umbrella.report import format_suggest
        result = run_suggest(
            model_name=args.model,
            vram_gb=args.vram,
            context_length=args.context,
            quality_mode=args.quality,
        )
        print(format_suggest(result, fmt))

    elif args.command == "validate":
        from umbrella.actions.validate import run_validate
        from umbrella.report import format_validate
        result = run_validate(
            bits=args.bits,
            seq_len=args.seq_len,
            synthetic=args.synthetic,
            backend=args.backend,
        )
        print(format_validate(result, fmt))

    elif args.command == "compress":
        from umbrella.actions.compress import run_compress
        from umbrella.report import format_compress
        result = run_compress(
            bits=args.bits,
            backend=args.backend,
            model_path=args.model,
            output_path=args.out_path,
        )
        print(format_compress(result, fmt))

    elif args.command == "autotune":
        from umbrella.actions.autotune import run_autotune
        from umbrella.report import format_autotune
        result = run_autotune(
            model_name=args.model,
            vram_gb=args.vram,
            context_length=args.context,
            quality_mode=args.quality,
        )
        print(format_autotune(result, fmt))

    elif args.command == "serve":
        from umbrella.api import create_app
        app = create_app(enable_mcp=args.mcp)
        print(f"  🌂 Umbrella API starting on http://{args.host}:{args.port}", file=sys.stderr)
        if args.mcp:
            print(f"  🔌 MCP endpoint: http://{args.host}:{args.port}/mcp", file=sys.stderr)
        print(f"  📖 Docs: http://{args.host}:{args.port}/docs\n", file=sys.stderr)
        try:
            import uvicorn
            uvicorn.run(app, host=args.host, port=args.port)
        except ImportError:
            print("  ❌ uvicorn not installed. Run: pip install uvicorn", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
