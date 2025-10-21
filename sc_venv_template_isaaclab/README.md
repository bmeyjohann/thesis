Container-Friendly IsaacLab Venv Template
=========================================

This template bootstraps a lightweight Python virtual environment **inside the Isaac Sim 5.0.0 container** without disturbing the Omniverse packages that ship with the image.

Key characteristics:
- Uses `/isaac-sim/python.sh` by default (override with `PYTHON_BIN=/path/to/python`).
- Creates the venv with `--system-site-packages` so Isaac Sim, Omniverse Kit, and CUDA wheels remain visible.
- Installs only the small set of extras listed in `requirements.txt`, plus editable copies of `IsaacLab` and `fasttd3` when those source trees are present.

Typical workflow (run inside the container, e.g. through `apptainer exec --nv ... /bin/bash -lc "<cmd>"`):

```bash
cd /workspace             # bind-mounted repository root
bash sc_venv_template_isaaclab/setup.sh
source sc_venv_template_isaaclab/activate.sh
python fasttd3/fast_sac/train.py --env_name Isaac-Lift-Cube-Franka-v0 ...
```

Environment variables recognised during setup:
- `PYTHON_BIN` – set to a different interpreter if needed (defaults to `/isaac-sim/python.sh`).
- `ISAACLAB_PATH` – path to the IsaacLab checkout to install in editable mode (defaults to `../IsaacLab`).
- `FASTTD3_PATH` – path to the fasttd3 sources (defaults to `../fasttd3`).

`requirements.txt` is intentionally minimal; add dependencies here only when they are **not** already supplied by Isaac Sim. If you need to install an extra package with overlapping transitive requirements, prefer `pip install --no-deps` and rely on the container’s versions.

Helper scripts:
- `activate.sh` – source this to enter the venv (must be sourced, not executed).
- `setup.sh` – creates/refreshed the venv and installs extras.
- `create_python_for_vscode.sh` – writes a small wrapper that launches the venv interpreter; useful for VS Code’s Python path picker.
- `create_kernel.sh` – optional helper to register a Jupyter kernel that activates the venv before launching `ipykernel`.
