"""
Central configuration for the LLM kernel optimizer.

All paths and tunable constants live here so the rest of the code
has a single place to look.
"""

import os
from pathlib import Path

# ── Repository layout ─────────────────────────────────────────────────────────
OPTIMIZER_ROOT   = Path(__file__).parent.resolve()          # llm-optimizer/
REPO_ROOT        = OPTIMIZER_ROOT.parent                    # amd-slingshot/

# MGPUSim paths
FIR_BENCHMARK_DIR = REPO_ROOT / "mgpusim" / "amd" / "benchmarks" / "heteromark" / "fir"
FIR_SAMPLE_DIR    = REPO_ROOT / "mgpusim" / "amd" / "samples" / "fir"
FIR_BINARY        = FIR_SAMPLE_DIR / "fir"

# Kernel files — the optimizer writes new kernels here; Go embeds kernels.hsaco
KERNEL_CL    = FIR_BENCHMARK_DIR / "kernels.cl"     # source written before compile
KERNEL_HSACO = FIR_BENCHMARK_DIR / "kernels.hsaco"  # output of compiler API

# Baseline and iteration kernels
KERNELS_DIR      = OPTIMIZER_ROOT / "kernels"
BASELINE_KERNEL  = KERNELS_DIR / "baseline_fir.cl"

# History / output directories
HISTORY_DIR   = OPTIMIZER_ROOT / "history"
METRICS_DIR   = HISTORY_DIR / "metrics"    # per-iteration metrics CSVs + sqlite3
LOGS_DIR      = HISTORY_DIR / "logs"       # per-run log files
HISTORY_JSON  = HISTORY_DIR / "history.json"
BEST_KERNEL   = HISTORY_DIR / "best_kernel.cl"

# Metrics CSV produced by the simulator (live, overwritten each run)
FIR_METRICS_CSV = HISTORY_DIR / "metrics.csv"

# ── API credentials ────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# ── LLM settings ──────────────────────────────────────────────────────────────
LLM_MODEL   = os.environ.get("LLM_MODEL",   "openai/gpt-oss-120b")
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.3"))

# ── Remote compiler API ───────────────────────────────────────────────────────
COMPILER_API_URL = os.environ.get("COMPILER_API_URL", "http://40.192.96.193:8000")

# ── Simulator settings ────────────────────────────────────────────────────────
SIMULATOR_API_URL = os.environ.get("SIMULATOR_API_URL", "http://localhost:8001")

AMDGPU_CPU         = os.environ.get("AMDGPU_CPU", "gfx803")   # GCN3 target
FIR_LENGTH         = int(os.environ.get("FIR_LENGTH", "8"))
SIM_TIMEOUT        = int(os.environ.get("SIM_TIMEOUT", "120"))  # seconds
SIMULATION_FLAGS   = [
    "-timing",
    "--report-all",
    "-disable-rtm",
]

# ── Optimizer loop settings ───────────────────────────────────────────────────
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "10"))

# ── Create directories on import ──────────────────────────────────────────────
HISTORY_DIR.mkdir(exist_ok=True)
KERNELS_DIR.mkdir(exist_ok=True)
METRICS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ── Architecture hints injected into every LLM system prompt ─────────────────
GCN3_HINTS = """
AMD GCN3 Architecture constraints relevant for kernel optimization:
- Wavefront size: 64 threads
- Max 256 work-items per work-group
- LDS (Local Data Store) per CU: 64 KB
- 4 SIMD units per CU, each 16-wide → 1 wavefront/SIMD
- Register file: 256 VGPRs per SIMD, 104 SGPRs per wavefront
- Occupancy drops sharply above 64 VGPRs/wavefront
- Coalesced global memory access: 64-byte cache lines (L1), 64-byte (L2)
- Prefer __local memory to reduce global traffic
- Avoid branch divergence within a wavefront

MGPUSim simulator limitations (HARD CONSTRAINTS — violating these causes a simulator panic):
- Do NOT use 64-bit scalar operations: S_LSHR_B64, S_LSHL_B64, S_AND_B64, S_OR_B64 etc.
  These are triggered by size_t arithmetic, ulong, pointer differences, or 64-bit indices.
- Use ONLY 32-bit types for indices: uint not size_t, int not ptrdiff_t.
- Do NOT use get_global_offset() — it emits unimplemented scalar instructions.
- Do NOT use printf() — not supported in MGPUSim.
- Avoid mul_hi(), mad_hi() on 64-bit types.
- All array indices and loop counters MUST be uint or int (32-bit), never ulong/size_t.
"""
