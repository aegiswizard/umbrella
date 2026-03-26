"""
Umbrella 🌂 — Report Formatter
Human-readable Markdown reports and machine-readable JSON for every mode.
"""

import json
from .schemas import DoctorResult, SuggestResult, ValidateResult, CompressResult, AutotuneResult

DIV = "─" * 56


def _md_section(title: str) -> str:
    return f"\n## {title}\n"


def format_doctor(r: DoctorResult, fmt: str = "text") -> str:
    if fmt == "json":
        return r.to_json()
    lines = [
        "",
        f"🌂 {DIV}",
        "   UMBRELLA — DOCTOR REPORT",
        f"🌂 {DIV}",
        "",
        f"   Verdict: {r.verdict}",
        "",
        "   🖥️  SYSTEM",
        f"       Python:     {r.python_version}",
        f"       RAM:        {r.ram_gb} GB",
        f"       PyTorch:    {'✅ ' + r.pytorch_version if r.pytorch_available else '❌ Not installed'}",
        f"       CUDA:       {'✅ ' + r.cuda_version + ' — ' + r.cuda_device_name + f' ({r.vram_gb} GB)' if r.cuda_available else '❌ Not available'}",
        f"       TurboQuant: {'✅ Installed' if r.turboquant_core else '❌ Not installed  (pip install turboquant)'}",
        f"       llama.cpp:  {'⚠️  EXPERIMENTAL — detected' if r.llamacpp_available else '—  Not found'}",
        "",
    ]
    if r.capabilities:
        lines.append("   ✅  AVAILABLE ON THIS MACHINE")
        for c in r.capabilities:
            lines.append(f"       · {c}")
        lines.append("")
    if r.limitations:
        lines.append("   ⚠️   LIMITATIONS")
        for l in r.limitations:
            lines.append(f"       · {l}")
        lines.append("")
    lines += [
        f"   Recommended backend: {r.recommended_backend}",
        f"   Full validation:     {'Yes' if r.full_validation_possible else 'No — CUDA + turboquant required'}",
        f"   Synthetic sim:       Yes — always available",
        "",
        f"🌂 {DIV}",
        "",
    ]
    return "\n".join(lines)


def format_suggest(r: SuggestResult, fmt: str = "text") -> str:
    if fmt == "json":
        return r.to_json()
    fits_icon = "✅" if r.fits_in_budget else "⚠️ "
    lines = [
        "",
        f"🌂 {DIV}",
        "   UMBRELLA — SUGGEST REPORT",
        f"🌂 {DIV}",
        "",
        f"   Model:          {r.model_name}",
        f"   VRAM budget:    {r.vram_gb} GB",
        f"   Context length: {r.context_length:,} tokens",
        f"   Quality mode:   {r.quality_mode}",
        "",
        "   🎯  RECOMMENDATION",
        f"       Bits:               {r.recommended_bits}-bit",
        f"       Compression ratio:  {r.expected_compression_ratio:.2f}x",
        f"       KV size baseline:   {r.baseline_kv_memory_gb:.3f} GB (FP16)",
        f"       KV size compressed: {r.compressed_kv_memory_gb:.3f} GB",
        f"       Memory saved:       {r.memory_saved_gb:.3f} GB",
        f"       Fits in budget:     {fits_icon} {'Yes' if r.fits_in_budget else 'No — see warnings'}",
        f"       Attention fidelity: ~{r.attention_fidelity_estimate*100:.2f}%",
        f"       Fallback option:    {r.fallback_bits}-bit",
        "",
    ]
    if r.reasoning:
        lines.append("   🧠  REASONING")
        for i, line in enumerate(r.reasoning, 1):
            lines.append(f"       {i}. {line}")
        lines.append("")
    if r.warnings:
        lines.append("   ⚠️   WARNINGS")
        for w in r.warnings:
            lines.append(f"       · {w}")
        lines.append("")
    if r.next_steps:
        lines.append("   ➜   NEXT STEPS")
        for s in r.next_steps:
            lines.append(f"       {s}")
        lines.append("")
    lines += [f"🌂 {DIV}", ""]
    return "\n".join(lines)


def format_validate(r: ValidateResult, fmt: str = "text") -> str:
    if fmt == "json":
        return r.to_json()
    status_icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌", "SKIPPED": "—"}.get(r.status, "?")
    lines = [
        "",
        f"🌂 {DIV}",
        "   UMBRELLA — VALIDATION REPORT",
        f"🌂 {DIV}",
        "",
        f"   {status_icon}  {r.status}  ({r.mode.upper()} mode)",
        f"   Backend: {r.backend}   Bits: {r.bits}",
        "",
    ]
    if r.metrics:
        m = r.metrics
        lines += [
            "   📊  METRICS",
            f"       Baseline memory:    {m.baseline_memory_mb:.2f} MB",
            f"       Compressed memory:  {m.compressed_memory_mb:.2f} MB",
            f"       Compression ratio:  {m.compression_ratio:.3f}x",
            f"       Attention similarity: {m.attention_similarity*100:.4f}%",
            f"       Top-k preservation:   {m.topk_preservation*100:.4f}%",
            f"       Sequence length:    {m.sequence_length}",
            f"       Heads × dim:        {m.num_heads} × {m.head_dim}",
            "",
        ]
    lines += [
        "   📋  VERDICT",
        f"       {r.verdict}",
        "",
    ]
    if r.detail:
        lines += ["   ℹ️   DETAIL", f"       {r.detail[:200]}", ""]
    if r.warnings:
        lines.append("   ⚠️   WARNINGS")
        for w in r.warnings:
            lines.append(f"       · {w}")
        lines.append("")
    lines += [
        f"   Hardware: {r.hardware_used}",
        "",
        f"🌂 {DIV}",
        "",
    ]
    return "\n".join(lines)


def format_compress(r: CompressResult, fmt: str = "text") -> str:
    if fmt == "json":
        return r.to_json()
    icon = "✅" if r.status == "success" else "❌"
    lines = [
        "",
        f"🌂 {DIV}",
        "   UMBRELLA — COMPRESS REPORT",
        f"🌂 {DIV}",
        "",
        f"   {icon}  {r.status.upper()}",
        f"   Backend: {r.backend}   Bits: {r.bits}",
        "",
    ]
    if r.compression_ratio:
        lines += [
            "   📦  RESULT",
            f"       Compression ratio:  {r.compression_ratio:.3f}x",
            f"       Before:             {r.memory_before_mb:.2f} MB" if r.memory_before_mb else "",
            f"       After:              {r.memory_after_mb:.2f} MB"  if r.memory_after_mb  else "",
            f"       Quality verified:   {'Yes' if r.quality_verified else 'No'}",
            "",
        ]
    if r.output_path:
        lines += [f"   Output: {r.output_path}", ""]
    if r.detail:
        lines += ["   ℹ️   DETAIL", f"       {r.detail[:300]}", ""]
    if r.warnings:
        lines.append("   ⚠️   WARNINGS")
        for w in r.warnings:
            lines.append(f"       · {w}")
        lines.append("")
    if r.deployment_preset:
        lines += [
            "   🚀  DEPLOYMENT PRESET",
            f"       Preset name: {r.deployment_preset.get('preset_name', '')}",
            f"       Use case:    {r.deployment_preset.get('use_case', '')}",
            f"       llama.cpp:   {r.deployment_preset.get('llama_cpp_arg', '')}",
            "",
        ]
    lines += [f"🌂 {DIV}", ""]
    return "\n".join(lines)


def format_autotune(r: AutotuneResult, fmt: str = "text") -> str:
    if fmt == "json":
        return r.to_json()
    fits_icon = "✅" if r.fits_in_budget else "⚠️ "
    lines = [
        "",
        f"🌂 {DIV}",
        "   UMBRELLA — AUTOTUNE REPORT",
        "   Don't just quantize. Prove it fits.",
        f"🌂 {DIV}",
        "",
        f"   Model:          {r.model_name}",
        f"   VRAM budget:    {r.vram_gb} GB",
        f"   Context length: {r.context_length:,} tokens",
        f"   Quality mode:   {r.quality_mode}",
        "",
        "   🎯  DECISION",
        f"       Chosen bits:     {r.chosen_bits}-bit",
        f"       Backend:         {r.chosen_backend}",
        f"       Savings:         {r.expected_savings_gb:.3f} GB",
        f"       Ratio:           {r.compression_ratio:.2f}x",
        f"       Fits in budget:  {fits_icon} {'Yes' if r.fits_in_budget else 'No'}",
        "",
    ]
    if r.reasoning:
        lines.append("   🧠  REASONING")
        for i, l in enumerate(r.reasoning, 1):
            lines.append(f"       {i}. {l}")
        lines.append("")
    lines.append("   ✅  VALIDATION PLAN")
    for s in r.validation_plan:
        lines.append(f"       · {s}")
    lines.append("")
    lines.append("   🔄  FALLBACK PLAN")
    for s in r.fallback_plan:
        lines.append(f"       · {s}")
    lines.append("")
    if r.warnings:
        lines.append("   ⚠️   WARNINGS")
        for w in r.warnings:
            lines.append(f"       · {w}")
        lines.append("")
    lines += [f"🌂 {DIV}", ""]
    return "\n".join(lines)
