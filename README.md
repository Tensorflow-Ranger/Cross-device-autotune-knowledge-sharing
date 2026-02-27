# Cross-Device Autotune Knowledge Sharing

GPU kernel optimization using an LLM feedback loop on MGPUSim. Automatically tunes OpenCL kernels for AMD GCN3 hardware by iteratively analyzing performance metrics and applying optimizations.

## Structure

- **llm-optimizer/** — The optimization agent. Runs a closed loop: rewrite → compile → simulate → analyze → plan → repeat
- **mgpusim/** — AMD GPU simulator. Executes kernels and generates performance metrics

## Quick Start

```bash
cd llm-optimizer
export GROQ_API_KEY=your_api_key
python3 orchestrator.py --iterations 5
```

See [llm-optimizer/README.md](llm-optimizer/README.md) for full details on setup, environment variables, and usage.

## How It Works

Each iteration:
1. **Rewriter** generates an optimized kernel using LLM guidance
2. **Compiler** compiles to HSACO and rebuilds the binary
3. **Simulator** runs the kernel in MGPUSim
4. **Metrics** are extracted and analyzed
5. **Profiler** identifies the bottleneck (memory, compute, latency)
6. **Planner** decides the next optimization strategy
7. Repeat with the new strategy

Results and metrics for each iteration are saved to `history/`.

## Requirements

- Go 1.20+
- Python 3.10+
- Groq API key (for LLM calls)
- OpenCL compiler (ROCm or clang-ocl)

## Project Goals

Reduce manual kernel tuning by automating the identify-optimize-test cycle using LLM agents that understand both GPU architecture and performance analytics.
