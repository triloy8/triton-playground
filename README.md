# Triton Playground

Experimental playground for writing and benchmarking Triton kernels with PyTorch orchestration.

## Layout

- `flashattention_2/` — FlashAttention-2 kernel (Triton + PyTorch wrapper).
- `weighted_sum/` — example kernel package (Triton + PyTorch wrapper).
- `pyproject.toml` — light uv-compatible project config.

## Quickstart (using uv)

- Install uv: https://docs.astral.sh/uv/
- Create and sync environment:
  - `uv venv -p 3.11`
  - `uv sync`  (installs deps and creates `uv.lock`)
- Run FlashAttention-2:
  - `uv run flashattention-2`
  - Or directly: `uv run python -m triton_playground.flashattention_2`
- Run weighted-sum demo:
  - `uv run weighted-sum`
  - Or directly: `uv run python -m triton_playground.weighted_sum`


Notes:
- The `weighted-sum` entrypoint uses the Triton-backed autograd Function when CUDA/Triton are available; otherwise it falls back to a pure PyTorch implementation on CPU.
- Add new kernels as sibling packages (one folder per kernel). Give each a `main()` in a dedicated script file and register it in `pyproject.toml` under `[project.scripts]`.

## Add a New Kernel

1. Create a new package directory, e.g. `softmax/` with your Triton kernel and a small PyTorch wrapper.
2. Create a `<kernel>/<kernel>.py` file with a `main()` function to drive your demo.
3. Register a console script in `pyproject.toml`:
   
   ```toml
   [project.scripts]
   my-kernel = "my_kernel.my_kernel:main"
   ```
4. Optionally expose a small API from your package for reuse.

## Development

- Lint: `uv run ruff check .`
- Test: `uv run pytest -q`
