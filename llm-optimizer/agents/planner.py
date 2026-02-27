"""
Agent 3 — Planner
=================
Reads the bottleneck JSON from the Profiler and the history of past strategies,
then decides the next concrete optimisation to try.

Returns a plain-English strategy instruction for the Rewriter Agent.
"""

import json
from pathlib import Path
from typing import Any

from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set.")
        _client = OpenAI(
            api_key=config.GROQ_API_KEY,
            base_url=config.GROQ_BASE_URL,
        )
    return _client


SYSTEM_PROMPT = f"""You are an expert GPU optimisation strategist for AMD GCN3 kernels.

You receive:
  1. A bottleneck JSON describing the current performance problem.
  2. A history list of strategies previously tried, each with whether it improved
     kernel_time_s or not (relative change shown).

Your job: decide the single best next optimisation strategy to try.

Rules:
  - Output ONLY the strategy instruction as 2–5 sentences of clear, actionable text.
  - Be specific: name the exact technique, which data structures to move to LDS, 
    what block size to use, etc.
  - Do NOT repeat a strategy that was tried and made no improvement.
  - Do NOT output JSON, bullet points, markdown, or headers — just plain prose.

{config.GCN3_HINTS}

Strategy decision logic:
  memory_bound   → try LDS tiling for coeff[], then input[] prefetch via __local,
                   then float4 vectorised loads, then work-group size tuning
  compute_bound  → try loop unrolling (#pragma unroll N), SIMD widening,
                   instruction-level parallelism via temporary accumulators
  underutilised  → increase work-group size (256 threads), increase problem size,
                   check HiddenGlobalOffset calculations
  latency_bound  → software prefetch of next iteration's data into registers,
                   overlap computation with memory via split accumulation loops
  unknown        → start with the most universally beneficial: LDS tiling for coeff[]
"""


def plan(bottleneck: dict[str, Any], history: list[dict[str, Any]]) -> str:
    """
    Given the current bottleneck and the history of attempts, return
    the next strategy as a plain-English string for the Rewriter.
    """
    history_text = _format_history(history)

    # Surface the most recent simulator panic (if any) as a hard constraint
    latest_panic = ""
    if history:
        latest_panic = history[-1].get("sim_panic", "")

    panic_section = ""
    if latest_panic:
        panic_section = (
            f"## CRITICAL — last kernel caused a simulator PANIC\n\n"
            f"  {latest_panic}\n\n"
            "The kernel triggered an unimplemented instruction in MGPUSim.\n"
            "The next strategy MUST avoid 64-bit scalar operations entirely.\n"
            "Require the Rewriter to use ONLY 32-bit types (uint/int) for all "
            "indices — absolutely no size_t, ulong, uint64_t, or pointer arithmetic.\n\n"
        )

    user_message = (
        f"{panic_section}"
        f"## Current bottleneck\n\n{json.dumps(bottleneck, indent=2, default=str)}\n\n"
        f"## History of strategies tried\n\n{history_text}\n\n"
        "What single optimisation strategy should the Rewriter attempt next? "
        "Be specific and actionable."
    )

    response = _get_client().chat.completions.create(
        model=config.LLM_MODEL,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    return response.choices[0].message.content.strip()


def _format_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "None — this is the first iteration."

    lines = []
    for i, entry in enumerate(history, 1):
        strategy  = entry.get("strategy",  "unknown")
        delta     = entry.get("time_delta_pct")
        bottleneck = entry.get("bound_by",  "unknown")
        if delta is not None:
            sign = "+" if delta > 0 else ""
            change = f"kernel_time changed {sign}{delta:.1f}%"
        else:
            change = "no timing comparison available"
        panic = entry.get("sim_panic", "")
        panic_note = f"  ⚠ SIMULATOR PANIC: {panic}" if panic else ""
        lines.append(f"{i}. [{bottleneck}] {strategy}  →  {change}{panic_note}")

    return "\n".join(lines)
