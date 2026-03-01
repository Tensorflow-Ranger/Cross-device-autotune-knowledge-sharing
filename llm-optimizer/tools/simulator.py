"""
Runs the MGPUSim ./fir benchmark and returns the resulting metrics.

Resolution order
────────────────
1. HTTP Simulator API  ( http://localhost:8001/simulate — simulator_api.py )
   Start it once with:  bash start_simulator.sh
2. Local subprocess    ( runs ./fir directly, no server needed )

Environment knobs (all optional):
    FIR_LENGTH          – number of taps / input samples (default 64 for speed)
    SIM_DISABLE_RTM     – set to "0" to keep RTM enabled  (default "1" = disabled)
    SIM_MAX_INST        – cap instruction count to limit sim time, e.g. "1000000"
    SIM_TIMEOUT         – seconds before subprocess is killed (default 120)
    SIMULATOR_API_URL   – override API base URL (default http://localhost:8001)
"""

import glob
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from tools.sqlite_to_csv import find_latest_sqlite, sqlite_to_csv

log = logging.getLogger(__name__)

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


class SimulatorError(Exception):
    """Raised when the simulator binary is absent or fundamentally broken."""


@dataclass
class SimulateResult:
    success:     bool
    returncode:  int   = 0
    stdout:      str   = ""
    stderr:      str   = ""
    elapsed_s:   float = 0.0
    metrics_csv: Path  = field(default=None)   # type: ignore[assignment]
    method:      str   = ""     # "api" | "local" | "fallback"
    is_fallback: bool  = False  # True when synthetic metrics were used
    db_path:     Path  = field(default=None)   # type: ignore[assignment]
    sim_panic:   str   = ""    # non-empty if simulator crashed with unimplemented opcode etc.


# ── public entry point ────────────────────────────────────────────────────────

def run_simulation(length: int | None = None, timeout: int | None = None, fallback: bool = True) -> SimulateResult:
    """
    Run the FIR benchmark and return a SimulateResult.

    Tries the HTTP simulator API first; falls back to running the binary
    directly as a subprocess. If both fail and fallback=True, returns
    synthetic metrics so the orchestrator can continue.

    On success, result.metrics_csv points to an existing, readable CSV file.
    Raises SimulatorError if the binary is missing entirely.
    """
    binary = config.FIR_BINARY
    if not binary.exists():
        raise SimulatorError(
            f"FIR binary not found at {binary}.\n"
            "Build it with:\n"
            "  cd mgpusim && go build -o amd/samples/fir/fir ./amd/samples/fir"
        )

    if _api_is_reachable():
        log.info("[simulator] Using HTTP simulator API")
        print(f"[simulator] Simulator API reachable — routing via API")
        result = _simulate_via_api(length=length, timeout=timeout)
        if result.success:
            return result
        log.warning("[simulator] API attempt failed: %s", result.stderr[:200])
        print(f"[simulator] API failed, falling back to local binary")

    result = _simulate_local(length=length, timeout=timeout)
    if result.success:
        return result

    log.error("[simulator] Local simulation failed (rc=%d): %s",
              result.returncode, result.stderr[:300])

    # If the kernel triggered an unimplemented opcode, do NOT swallow it with
    # synthetic metrics — propagate the real failure so the orchestrator can
    # pass it to the planner for a corrected kernel.
    if result.sim_panic:
        log.error("[simulator] Simulator PANIC detected — skipping fallback so "
                  "the orchestrator can recover: %s", result.sim_panic)
        return result

    # Both failed: use fallback synthetic metrics if enabled
    if fallback:
        log.warning("[simulator] FALLBACK: writing synthetic metrics — results will NOT reflect kernel changes!")
        print("[simulator] *** FALLBACK MODE: using synthetic metrics — kernel changes have NO effect ***")
        return _simulate_fallback(length=length)
    else:
        return result


# ── HTTP API path ─────────────────────────────────────────────────────────────

def _api_url() -> str:
    return config.SIMULATOR_API_URL.rstrip("/")


def _api_is_reachable() -> bool:
    if not _REQUESTS_AVAILABLE:
        return False
    try:
        r = _requests.get(f"{_api_url()}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _simulate_via_api(length: int | None, timeout: int | None) -> SimulateResult:
    params: dict = {}
    if length  is not None:
        params["length"]  = length
    if timeout is not None:
        params["timeout"] = timeout

    t0_wall = time.time()   # wall-clock used for min_mtime filtering
    t0 = time.monotonic()
    try:
        # Give HTTP request buffer time beyond the sim timeout
        http_timeout = (timeout or config.SIM_TIMEOUT or 30) + 30
        resp = _requests.post(
            f"{_api_url()}/simulate",
            params=params,
            timeout=http_timeout,
        )
    except Exception:
        # API unreachable mid-call: fall back to local
        return _simulate_local(length=length, timeout=timeout)

    elapsed = time.monotonic() - t0

    if resp.status_code == 504:
        detail = resp.json().get("detail", {})
        return SimulateResult(
            success=False,
            returncode=-1,
            stderr=(
                f"Simulation timed out after {detail.get('elapsed', '?')}s.\n"
                f"Hint: {detail.get('hint', '')}"
            ),
            elapsed_s=elapsed,
            method="api",
        )

    if resp.status_code != 200:
        detail = resp.json().get("detail", str(resp.text[:400]))
        return SimulateResult(
            success=False,
            returncode=resp.status_code,
            stderr=str(detail),
            elapsed_s=elapsed,
            method="api",
        )

    # API ran ./fir in FIR_SAMPLE_DIR — convert latest sqlite3 → metrics.csv
    metrics_path = config.FIR_METRICS_CSV
    db_path = find_latest_sqlite(config.FIR_SAMPLE_DIR, min_mtime=t0_wall)
    if db_path is None:
        log.error("[simulator/api] No fresh sqlite3 found after simulation (t0=%.3f)", t0_wall)
        return SimulateResult(
            success=False,
            returncode=0,
            stderr=f"API returned 200 but no fresh akita_sim_*.sqlite3 found in {config.FIR_SAMPLE_DIR}.",
            elapsed_s=elapsed,
            method="api",
        )

    log.info("[simulator/api] Converting %s → metrics.csv", db_path.name)
    print(f"[simulator] Converting {db_path.name} → metrics.csv")
    try:
        sqlite_to_csv(db_path, metrics_path)
    except ValueError as exc:
        log.error("[simulator/api] sqlite_to_csv failed: %s", exc)
        return SimulateResult(
            success=False,
            returncode=0,
            stderr=f"sqlite_to_csv failed: {exc}",
            elapsed_s=elapsed,
            method="api",
        )

    return SimulateResult(
        success=True,
        returncode=0,
        elapsed_s=elapsed,
        metrics_csv=metrics_path,
        method="api",
        db_path=db_path,
    )


# ── local subprocess path ─────────────────────────────────────────────────────

def _simulate_local(length: int | None, timeout: int | None) -> SimulateResult:
    fir_length = length  if length  is not None else config.FIR_LENGTH
    timeout_s  = timeout if timeout is not None else config.SIM_TIMEOUT

    cmd = [
        str(config.FIR_BINARY),
        f"-length={fir_length}",
        *config.SIMULATION_FLAGS,
    ]

    log.info("[simulator/local] cmd: %s", " ".join(cmd))
    print(f"[simulator] Running: {' '.join(cmd)}")
    print(f"[simulator] cwd: {config.FIR_SAMPLE_DIR}  timeout: {timeout_s}s")

    t0_wall = time.time()   # wall-clock for min_mtime filtering
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(config.FIR_SAMPLE_DIR),
            timeout=timeout_s if timeout_s > 0 else None,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        return SimulateResult(
            success=False,
            returncode=-1,
            stderr=(
                f"Simulation timed out after {timeout_s}s "
                f"(FIR_LENGTH={fir_length}).\n"
                "Reduce FIR_LENGTH or set SIM_MAX_INST to cap instruction count.\n"
                "Example:  export FIR_LENGTH=64  or  export SIM_MAX_INST=500000"
            ),
            elapsed_s=elapsed,
            method="local",
        )

    elapsed = time.monotonic() - t0

    # ── Log raw simulator output ───────────────────────────────────────────────
    if result.stdout.strip():
        log.info("[simulator/local] stdout:\n%s", result.stdout)
        print(f"[simulator] stdout:\n{result.stdout.strip()}")
    if result.stderr.strip():
        log.info("[simulator/local] stderr:\n%s", result.stderr)
        print(f"[simulator] stderr:\n{result.stderr.strip()}")

    # ── Detect simulator panic / unimplemented opcode ────────────────────────
    import re as _re
    panic_match = _re.search(
        r"(Opcode \d+ for \w+ format is not implemented|Panic:.*|not implemented.*)",
        result.stderr
    )
    sim_panic = panic_match.group(0) if panic_match else ""
    if sim_panic:
        log.error("[simulator/local] PANIC detected: %s", sim_panic)
        print(f"[simulator] PANIC: {sim_panic}")

    if result.returncode != 0:
        log.error("[simulator/local] non-zero exit %d", result.returncode)
        return SimulateResult(
            success=False,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            elapsed_s=elapsed,
            method="local",
            sim_panic=sim_panic,
        )

    # ── Find fresh sqlite3 (must be newer than t0_wall) ───────────────────────
    metrics_path = config.FIR_METRICS_CSV
    db_path = find_latest_sqlite(config.FIR_SAMPLE_DIR, min_mtime=t0_wall)
    if db_path is None:
        log.error("[simulator/local] No fresh sqlite3 found (t0_wall=%.3f)", t0_wall)
        # List what IS there for diagnostics
        existing = list(config.FIR_SAMPLE_DIR.glob("akita_sim_*.sqlite3"))
        log.error("[simulator/local] Existing sqlite3 files: %s", existing)
        print(f"[simulator] ERROR: no fresh akita_sim_*.sqlite3 found.")
        print(f"[simulator] Existing sqlite3 files: {[p.name for p in existing]}")
        return SimulateResult(
            success=False,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=(
                f"Simulation exited 0 but no fresh akita_sim_*.sqlite3 found in "
                f"{config.FIR_SAMPLE_DIR}.\n" + result.stderr
            ),
            elapsed_s=elapsed,
            method="local",
        )

    # ── Convert sqlite3 → metrics.csv ────────────────────────────────────────
    log.info("[simulator/local] Converting %s → metrics.csv", db_path.name)
    print(f"[simulator] Converting {db_path.name} → metrics.csv")
    try:
        sqlite_to_csv(db_path, metrics_path)
    except ValueError as exc:
        log.error("[simulator/local] sqlite_to_csv failed: %s", exc)
        return SimulateResult(
            success=False,
            returncode=0,
            stdout=result.stdout,
            stderr=f"sqlite_to_csv failed: {exc}",
            elapsed_s=elapsed,
            method="local",
        )

    return SimulateResult(
        success=True,
        returncode=0,
        stdout=result.stdout,
        stderr=result.stderr,
        elapsed_s=elapsed,
        metrics_csv=metrics_path,
        method="local",
        db_path=db_path,
    )


# ── fallback mode (synthetic metrics when simulator is broken) ─────────────────

def _simulate_fallback(length: int | None) -> SimulateResult:
    """
    Return synthetic metrics when the actual simulator times out/fails.
    Allows the orchestrator loop to continue even if ./fir is broken.
    """
    fir_length = length if length is not None else config.FIR_LENGTH

    # Write a synthetic metrics.csv so metrics_parser can read it
    csv_path = config.FIR_METRICS_CSV
    _write_synthetic_metrics(csv_path, fir_length)

    return SimulateResult(
        success=True,
        returncode=0,
        stdout="",
        stderr="[FALLBACK MODE] Simulator binary hangs; using synthetic metrics.",
        elapsed_s=0.0,
        metrics_csv=csv_path,
        method="fallback",
        is_fallback=True,
    )


def _write_synthetic_metrics(csv_path: Path, fir_length: int) -> None:
    """Generate a synthetic metrics.csv for testing when the binary is broken."""
    import csv

    # Baseline-like metrics for FIR filter at given length
    # These are reasonable approximate values
    metrics_data = [
        ("Driver", "kernel_time", fir_length * 0.5e-6, "s"),
        ("GPU[0].SA[0].CU[0]", "cu_inst_count", fir_length * 100, ""),
        ("GPU[0].SA[0].CU[0]", "cu_CPI", 4.5, ""),
        ("GPU[0].SA[0].CU[0]", "simd_inst_count", fir_length * 50, ""),
        ("GPU[0].SA[0].CU[0]", "simd_CPI", 5.0, ""),
        ("GPU[0].L1Cache[0]", "read_hit_count", fir_length * 80, ""),
        ("GPU[0].L1Cache[0]", "read_miss_count", fir_length * 20, ""),
        ("GPU[0].L2Cache", "read_hit_count", fir_length * 15, ""),
        ("GPU[0].L2Cache", "read_miss_count", fir_length * 5, ""),
        ("GPU[0].DRAM[0]", "read_trans_count", fir_length * 2, ""),
        ("GPU[0].DRAM[0]", "write_trans_count", fir_length * 1, ""),
        ("GPU[0].DRAM[0]", "read_size", fir_length * 256, "bytes"),
        ("GPU[0].DRAM[0]", "write_size", fir_length * 128, "bytes"),
        ("GPU[0].DRAM[0]", "read_avg_latency", 1.5e-7, "s"),
        ("GPU[0].L2ToDRAM", "read_trans_count", fir_length * 2, ""),
        ("GPU[0].L2ToDRAM", "write_trans_count", fir_length * 1, ""),
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Location", "What", "Value", "Unit"])
        for location, what, value, unit in metrics_data:
            writer.writerow([location, what, value, unit])
