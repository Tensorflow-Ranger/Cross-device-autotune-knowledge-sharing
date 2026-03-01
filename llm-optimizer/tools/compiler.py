"""
Compiler tool — turns an OpenCL kernel string into a runnable ./fir binary.

Pipeline
────────
1. Write new .cl source → KERNEL_CL     (mgpusim benchmark dir)
2. POST to remote Compiler API          → KERNEL_HSACO (validated ELF)
   Fallback: local clang-ocl / rocm-ocl if API unreachable
3. Rebuild Go ./fir binary              (embeds kernels.hsaco via //go:embed)
"""

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger(__name__)

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class CompileResult:
    success: bool
    method:  str  = ""    # "api" | "local" | "skipped"
    log:     str  = ""    # compiler stdout / error message
    elapsed_s: float = 0.0


# ── Isolated Go build environment ─────────────────────────────────────────────

_GO_CACHE  = config.OPTIMIZER_ROOT / ".go_cache"
_GO_TMP    = config.OPTIMIZER_ROOT / ".go_tmp"
_GOPATH    = config.OPTIMIZER_ROOT / ".gopath"

# Ensure Go work directories exist before any build
for _d in (_GO_CACHE, _GO_TMP, _GOPATH):
    _d.mkdir(parents=True, exist_ok=True)

_GO_ENV = {
    **os.environ,
    "GOCACHE": str(_GO_CACHE),
    "GOTMPDIR": str(_GO_TMP),
    "GOPATH":  str(_GOPATH),
}


# ── Public entry point ─────────────────────────────────────────────────────────

def compile_kernel(kernel_source: str, iteration: int | None = None) -> CompileResult:
    """
    Compile kernel_source (.cl) to .hsaco and rebuild the Go FIR binary.

    Steps:
      1. Save .cl to kernels/ directory (for history) and to KERNEL_CL.
      2. Compile via remote API (or local fallback).
      3. Rebuild Go binary so it embeds the new .hsaco.
    """
    t0 = time.perf_counter()

    # Back up existing .cl/.hsaco so we can restore them if the build fails
    _cl_backup: str | None = None
    _hsaco_backup_compile: bytes | None = None
    if config.KERNEL_CL.exists():
        _cl_backup = config.KERNEL_CL.read_text(encoding="utf-8")
    if config.KERNEL_HSACO.exists():
        _hsaco_backup_compile = config.KERNEL_HSACO.read_bytes()

    # 1. Save .cl sources
    config.KERNEL_CL.write_text(kernel_source, encoding="utf-8")
    log.info("[compiler] Wrote .cl → %s", config.KERNEL_CL)

    if iteration is not None:
        iter_cl = config.KERNELS_DIR / f"kernel_iter{iteration:03d}.cl"
        iter_cl.write_text(kernel_source, encoding="utf-8")
        log.debug("[compiler] Archived iteration kernel → %s", iter_cl)

    # 2. Compile .cl → .hsaco
    compile_result = _compile_api(kernel_source)
    if not compile_result.success:
        compile_result.elapsed_s = time.perf_counter() - t0
        return compile_result

    # 3. Rebuild Go binary (embeds the new .hsaco)
    build_result = _rebuild_go_binary_local()
    build_result.elapsed_s = time.perf_counter() - t0
    if not build_result.success:
        # Restore previous .cl and .hsaco so the binary on disk stays consistent
        # with the kernel source.  Without this, future simulator runs will panic
        # because the binary embeds the old .hsaco while kernels.cl shows the new
        # (broken) kernel.
        log.warning("[compiler] Go build failed — restoring previous kernels.cl/.hsaco")
        if _cl_backup is not None:
            config.KERNEL_CL.write_text(_cl_backup, encoding="utf-8")
            log.info("[compiler] Restored kernels.cl to previous version")
        if _hsaco_backup_compile is not None:
            config.KERNEL_HSACO.write_bytes(_hsaco_backup_compile)
            log.info("[compiler] Restored kernels.hsaco to previous version")
        return build_result

    log.info("[compiler] Done in %.1fs", build_result.elapsed_s)
    return CompileResult(
        success=True,
        method="api+gobuild",
        log=compile_result.log + "\n" + build_result.log,
        elapsed_s=build_result.elapsed_s,
    )


# ── Compilation via remote API ────────────────────────────────────────────────

def _compile_api(kernel_source: str) -> CompileResult:
    """POST the .cl source to the remote compiler API, validate the response."""
    if not _REQUESTS_AVAILABLE:
        return CompileResult(success=False, method="api",
                             log="'requests' package not installed.")

    url = f"{config.COMPILER_API_URL.rstrip('/')}/compile"
    log.info("[compiler/api] POST %s", url)

    # Back up current .hsaco in case we need to restore on failure
    _hsaco_backup: bytes | None = None
    if config.KERNEL_HSACO.exists():
        _hsaco_backup = config.KERNEL_HSACO.read_bytes()

    try:
        with config.KERNEL_CL.open("rb") as fh:
            resp = _requests.post(
                url,
                files={"file": ("kernels.cl", fh, "text/plain")},
                timeout=120,
            )
    except Exception as exc:
        if _hsaco_backup:
            config.KERNEL_HSACO.write_bytes(_hsaco_backup)
        return CompileResult(success=False, method="api",
                             log=f"Compiler API request failed: {exc}")

    if resp.status_code != 200:
        if _hsaco_backup:
            config.KERNEL_HSACO.write_bytes(_hsaco_backup)
        return CompileResult(
            success=False,
            method="api",
            log=f"Compiler API returned HTTP {resp.status_code}:\n{resp.text[:800]}",
        )

    # Validate the response is actually an ELF/HSACO binary, not a JSON error
    content = resp.content
    ELF_MAGIC = b'\x7fELF'
    if not content.startswith(ELF_MAGIC):
        if _hsaco_backup:
            config.KERNEL_HSACO.write_bytes(_hsaco_backup)
        preview = content[:300].decode('utf-8', errors='replace')
        return CompileResult(
            success=False,
            method="api",
            log=(
                f"Compiler API returned HTTP 200 but content is not a valid HSACO/ELF binary.\n"
                f"First 300 bytes: {preview}"
            ),
        )

    # Write .hsaco returned by the API
    config.KERNEL_HSACO.write_bytes(content)
    log.info("[compiler/api] Wrote %d bytes → %s", len(content), config.KERNEL_HSACO)
    return CompileResult(success=True, method="api",
                         log=f"Compiled OK ({len(content)} bytes)")


# ── Rebuild Go binary ──────────────────────────────────────────────────────────

def _rebuild_go_binary_local() -> CompileResult:
    """
    Rebuild the ./fir binary inside mgpusim/.
    The binary embeds kernels.hsaco via //go:embed, so the new .hsaco is
    automatically picked up when we rebuild.
    """
    mgpusim_root = config.REPO_ROOT / "mgpusim"
    output_bin   = config.FIR_BINARY
    package_path = "./amd/samples/fir"

    log.info("[compiler/go] Cleaning Go build cache ...")
    try:
        subprocess.run(
            ["go", "clean", "-cache"],
            cwd=mgpusim_root,
            env=_GO_ENV,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as exc:
        log.warning("[compiler/go] go clean -cache failed (non-fatal): %s", exc)

    log.info("[compiler/go] Building %s ...", package_path)
    try:
        proc = subprocess.run(
            ["go", "build", "-o", str(output_bin), package_path],
            cwd=mgpusim_root,
            env=_GO_ENV,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return CompileResult(success=False, method="gobuild",
                             log="go build timed out after 300s")
    except FileNotFoundError:
        return CompileResult(success=False, method="gobuild",
                             log="'go' binary not found — is Go installed and in PATH?")

    combined = (proc.stdout + "\n" + proc.stderr).strip()
    if proc.returncode != 0:
        log.error("[compiler/go] go build failed (rc=%d):\n%s", proc.returncode, combined)
        return CompileResult(success=False, method="gobuild", log=combined)

    log.info("[compiler/go] Built → %s", output_bin)
    return CompileResult(success=True, method="gobuild", log=combined or "OK")
