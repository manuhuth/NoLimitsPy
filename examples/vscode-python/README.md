# NoLimitsPy VS Code demo

A ready-to-open VS Code project for using NoLimits.jl from Python.

## Setup (once)

```bash
cd examples/vscode-python
python3 -m venv .venv
.venv/bin/pip install -e ../.. pandas ipykernel
```

Outside a clone of this repository, install from GitHub instead:

```bash
.venv/bin/pip install "git+https://github.com/manuhuth/NoLimitsPy" pandas ipykernel
```

## Use

1. Open THIS folder in VS Code (`code examples/vscode-python`).
2. The interpreter is preconfigured to `.venv` via `.vscode/settings.json`
   (VS Code may ask once to confirm).
3. Open `demo.py` and run it cell by cell with Shift+Enter (Python and Jupyter
   extensions required). Plots render inline in the Interactive Window.

The first Julia boot in a fresh venv resolves and precompiles the Julia
environment (minutes); afterwards sessions start in seconds. Keep the
Interactive Window alive between runs to avoid re-paying model compilation.
