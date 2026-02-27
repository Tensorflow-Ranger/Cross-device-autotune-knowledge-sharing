"""
Orchestrator — main optimization loop
======================================

Usage:
    python orchestrator.py [--iterations N] [--dry-run]

Flags:
    --iterations N   Override MAX_ITERATIONS from config.py
    --dry-run        Run simulation on the baseline kernel, print profiling
                     output, then exit without entering the LLM loop
    --resume PATH    Resume from a saved history JSON file
    --no-compile     Skip compilation (useful when ROCm is absent; LLMs still
                     run and .cl files are saved to kernels/)

Flow per iteration:
    1. Rewriter  → new kernel .cl
    2. Compiler  → .hsaco + rebuild binary   [skipped with --no-compile]
    3. Simulator → metrics.csv
    4. Profiler  → bottleneck JSON
    5. Planner   → strategy for next iter
    6. Persist history
"""

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# make imports work when run as a script
sys.path.insert(0, str(Path(__file__).parent))

import config
from agents import profiler, planner, rewriter
from tools  import compiler, simulator, metrics_parser


def main():
    args = _parse_args()

    # ── Set up logging (console + file) ──────────────────────────────────────────
    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = config.LOGS_DIR / f"run_{run_ts}.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("orchestrator")
    log.info("Log file: %s", log_path)

    log.info("="*60)
    log.info("  MGPUSim LLM Kernel Optimizer")
    log.info("  Model : %s", config.LLM_MODEL)
    log.info("  Target: %s  |  length=%s", config.AMDGPU_CPU, config.FIR_LENGTH)
    log.info("="*60)

    # ── load baseline kernel ──────────────────────────────────────────────────
    kernel_source = config.BASELINE_KERNEL.read_text()
    log.info("[init] Loaded baseline kernel from %s", config.BASELINE_KERNEL)

    # ── optionally resume from saved history ──────────────────────────────
    history: list[dict[str, Any]] = []
    if args.resume:
        history = json.loads(Path(args.resume).read_text())
        log.info("[init] Resumed %d past iterations from %s", len(history), args.resume)

    # ── dry-run: just profile the baseline ───────────────────────────────
    if args.dry_run:
        _dry_run(kernel_source, log)
        return

    best_time: float | None = None
    iterations = args.iterations or config.MAX_ITERATIONS
    strategy   = "No prior analysis available — start with the most impactful baseline optimisation."

    for iteration in range(1, iterations + 1):
        log.info("─"*60)
        log.info("  Iteration %d/%d", iteration, iterations)
        log.info("─"*60)

        # ── STEP 1 · Rewrite ───────────────────────────────────────────────
        log.info("[%d] Rewriter  → generating optimised kernel ...", iteration)
        try:
            new_kernel = rewriter.rewrite(kernel_source, strategy)
        except Exception as exc:
            log.error("[%d] Rewriter  ERROR: %s", iteration, exc)
            break
        log.debug("[%d] Rewriter  kernel (first 400 chars):\n%.400s", iteration, new_kernel)

        # ── STEP 2 · Compile ─────────────────────────────────────────────────
        if not args.no_compile:
            log.info("[%d] Compiler  → building .hsaco + fir binary ...", iteration)
            compile_result = compiler.compile_kernel(new_kernel, iteration)
            if not compile_result.success:
                log.error("[%d] Compiler  FAILED:\n%s", iteration, compile_result.log)
                log.warning("[%d] Saving .cl anyway — fix toolchain to proceed.", iteration)
                _save_cl(new_kernel, iteration)
                break
            log.info("[%d] Compiler  OK", iteration)
        else:
            _save_cl(new_kernel, iteration)
            log.info("[%d] Compiler  SKIPPED (--no-compile), .cl saved.", iteration)

        # ── STEP 3 · Simulate ────────────────────────────────────────────────
        sim_result = None
        if args.skip_sim:
            cached = config.FIR_METRICS_CSV
            if not cached.exists():
                log.error("[%d] Simulator SKIPPED but %s not found", iteration, cached)
                break
            log.info("[%d] Simulator SKIPPED (--skip-sim), using cached %s", iteration, cached.name)
            metrics_csv = cached
        else:
            log.info("[%d] Simulator → running ./fir ...", iteration)
            try:
                sim_result = simulator.run_simulation()
            except simulator.SimulatorError as exc:
                log.error("[%d] Simulator ERROR: %s", iteration, exc)
                break

            if not sim_result.success:
                log.error("[%d] Simulator FAILED (rc=%d)\n%s",
                          iteration, sim_result.returncode, sim_result.stderr)
                if sim_result.sim_panic:
                    # Kernel triggered an unimplemented instruction.
                    # Record the failure in history so the planner knows to
                    # generate a 32-bit-clean kernel next iteration.
                    log.warning("[%d] Panic recorded — planner will be instructed "
                                "to avoid 64-bit ops on next iteration.", iteration)
                    history_entry: dict[str, Any] = {
                        "iteration":     iteration,
                        "strategy":      strategy,
                        "kernel_source": new_kernel,
                        "metrics":       {},
                        "bottleneck":    {"bound_by": "sim_panic"},
                        "kernel_time_s": None,
                        "time_delta_pct": None,
                        "bound_by":      "sim_panic",
                        "is_fallback":   False,
                        "sim_method":    sim_result.method,
                        "db_file":       None,
                        "sim_panic":     sim_result.sim_panic,
                    }
                    history.append(history_entry)
                    _persist_history(history)
                    # Ask planner for a corrected strategy
                    try:
                        bottleneck = {"bound_by": "sim_panic",
                                      "bottleneck_summary": sim_result.sim_panic}
                        strategy = planner.plan(bottleneck, _trim_history(history))
                        log.info("[%d] Planner (panic recovery) → %s", iteration, strategy[:160])
                    except Exception as exc:
                        log.error("[%d] Planner ERROR during panic recovery: %s", iteration, exc)
                        break
                    kernel_source = new_kernel
                    continue
                break

            if sim_result.is_fallback:
                log.warning("[%d] *** FALLBACK METRICS — synthetic data, NOT from real simulation ***", iteration)
                log.warning("[%d] *** Kernel changes have NO effect on these numbers — fix the simulator! ***", iteration)

            if sim_result.sim_panic:
                log.error("[%d] *** SIMULATOR PANIC: %s ***", iteration, sim_result.sim_panic)
                log.error("[%d] The kernel triggered an unimplemented instruction. "
                          "This will be passed to the planner/rewriter.", iteration)

            log.info("[%d] Simulator OK  (wall=%.2fs  method=%s  db=%s)",
                     iteration, sim_result.elapsed_s, sim_result.method,
                     sim_result.db_path.name if sim_result.db_path else "none")
            metrics_csv = sim_result.metrics_csv

            # ── Archive per-iteration metrics CSV + sqlite3 ──────────────────
            iter_csv = config.METRICS_DIR / f"metrics_iter{iteration:03d}.csv"
            shutil.copy2(metrics_csv, iter_csv)
            log.info("[%d] Saved metrics snapshot → %s", iteration, iter_csv)

            if sim_result.db_path is not None:
                iter_db = config.METRICS_DIR / f"akita_iter{iteration:03d}.sqlite3"
                shutil.copy2(sim_result.db_path, iter_db)
                log.info("[%d] Archived sqlite3 → %s", iteration, iter_db)

        # ── STEP 4 · Parse metrics ──────────────────────────────────────────────
        metrics = metrics_parser.parse(metrics_csv)
        current_time = metrics.get("kernel_time_s")
        log.info("[%d] Metrics   kernel_time=%.4es  l1_miss=%s  l2_miss=%s  avg_cpi=%s",
                 iteration, current_time or 0,
                 metrics.get('l1_miss_rate'), metrics.get('l2_miss_rate'),
                 metrics.get('avg_cpi'))
        log.debug("[%d] Full metrics: %s", iteration, json.dumps(metrics, default=str))

        # track best
        if current_time is not None:
            if best_time is None or current_time < best_time:
                _save_best(new_kernel, iteration)
                best_time = current_time
                log.info("[%d] ★ New best: %.4es", iteration, best_time)

        # ── STEP 5 · Profile ───────────────────────────────────────────────────
        log.info("[%d] Profiler  → interpreting metrics ...", iteration)
        try:
            bottleneck = profiler.interpret(metrics)
        except Exception as exc:
            log.error("[%d] Profiler  ERROR: %s", iteration, exc)
            bottleneck = {"bound_by": "unknown", "bottleneck_summary": str(exc)}

        log.info("[%d] Profiler  bound_by=%s | %s",
                 iteration, bottleneck.get('bound_by'),
                 bottleneck.get('bottleneck_summary', '')[:120])

        # ── STEP 6 · Plan next strategy ───────────────────────────────────────
        # compute % change vs previous kernel_time
        time_delta_pct = None
        if history and history[-1].get("kernel_time_s") and current_time:
            prev = history[-1]["kernel_time_s"]
            time_delta_pct = ((current_time - prev) / prev) * 100.0

        history_entry: dict[str, Any] = {
            "iteration":     iteration,
            "strategy":      strategy,
            "kernel_source": new_kernel,
            "metrics":       metrics,
            "bottleneck":    bottleneck,
            "kernel_time_s": current_time,
            "time_delta_pct": time_delta_pct,
            "bound_by":      bottleneck.get("bound_by"),
            "is_fallback":   (sim_result.is_fallback if sim_result else False),
            "sim_method":    (sim_result.method      if sim_result else "skipped"),
            "db_file":       (sim_result.db_path.name if sim_result and sim_result.db_path else None),
            "sim_panic":     (sim_result.sim_panic    if sim_result else ""),
        }
        history.append(history_entry)
        _persist_history(history)

        log.info("[%d] Planner   → deciding next strategy ...", iteration)
        try:
            strategy = planner.plan(bottleneck, _trim_history(history))
        except Exception as exc:
            log.error("[%d] Planner   ERROR: %s", iteration, exc)
            break

        log.info("[%d] Planner   → %s", iteration, strategy[:160])
        kernel_source = new_kernel

    # ── summary ───────────────────────────────────────────────────────────────
    log.info("="*60)
    log.info("  Optimization complete — %d iteration(s)", len(history))
    if best_time is not None:
        log.info("  Best kernel_time : %.4es", best_time)
        log.info("  Best kernel saved: %s/best_kernel.cl", config.HISTORY_DIR)
    log.info("  Full history     : %s/history.json", config.HISTORY_DIR)
    log.info("  Per-iter metrics : %s/", config.METRICS_DIR)
    log.info("  Run log          : %s", log_path)
    log.info("="*60)


# ── helpers ───────────────────────────────────────────────────────────────────

def _dry_run(kernel_source: str, log: logging.Logger | None = None) -> None:
    _log = log or logging.getLogger("orchestrator")
    _log.info("[dry-run] Running baseline simulation ...")
    try:
        sim = simulator.run_simulation()
    except simulator.SimulatorError as exc:
        _log.error("[dry-run] ERROR: %s", exc)
        return

    if not sim.success:
        _log.error("[dry-run] Simulation failed:\n%s", sim.stderr)
        return

    if sim.is_fallback:
        _log.warning("[dry-run] WARNING: fallback synthetic metrics in use!")

    metrics = metrics_parser.parse(sim.metrics_csv)
    _log.info("[dry-run] Raw metrics:")
    for k, v in metrics.items():
        _log.info("  %-30s = %s", k, v)

    _log.info("[dry-run] Profiling ...")
    try:
        bottleneck = profiler.interpret(metrics)
        _log.info("[dry-run] Bottleneck:\n%s", json.dumps(bottleneck, indent=2))
    except Exception as exc:
        _log.error("[dry-run] Profiler error: %s", exc)


def _save_cl(source: str, iteration: int) -> None:
    path = config.KERNELS_DIR / f"kernel_iter{iteration:03d}.cl"
    path.write_text(source)


def _save_best(source: str, iteration: int) -> None:
    best_path = config.HISTORY_DIR / "best_kernel.cl"
    best_path.write_text(source)


def _persist_history(history: list[dict]) -> None:
    path = config.HISTORY_DIR / "history.json"
    # strip kernel source from JSON to keep file readable; it's already in kernels/
    compact = [
        {k: v for k, v in entry.items() if k != "kernel_source"}
        for entry in history
    ]
    path.write_text(json.dumps(compact, indent=2, default=str))


def _trim_history(history: list[dict]) -> list[dict]:
    """Return a condensed view for the planner prompt (avoid token overflow)."""
    return [
        {
            "strategy":       e.get("strategy", ""),
            "bound_by":       e.get("bound_by",  "unknown"),
            "kernel_time_s":  e.get("kernel_time_s"),
            "time_delta_pct": e.get("time_delta_pct"),
        }
        for e in history
    ]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MGPUSim LLM Kernel Optimizer")
    p.add_argument("--iterations", type=int, default=None,
                   help="Number of optimization iterations (default: from config.py)")
    p.add_argument("--dry-run", action="store_true",
                   help="Run baseline simulation + profiling only, then exit")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a history.json file to resume from")
    p.add_argument("--no-compile", action="store_true",
                   help="Skip compilation; only save .cl files and run LLM agents")
    p.add_argument("--skip-sim", action="store_true",
                   help="Skip the MGPUSim run and reuse the existing metrics.csv on disk")
    return p.parse_args()


if __name__ == "__main__":
    main()
