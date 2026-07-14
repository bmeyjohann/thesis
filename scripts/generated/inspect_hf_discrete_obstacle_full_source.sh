#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis

python - <<'PY'
import inspect
import mjlab.terrains as terrain_gen

src = inspect.getsource(terrain_gen.HfDiscreteObstaclesTerrainCfg.function)
for i, line in enumerate(src.splitlines(), 1):
    print(f"{i:04d}: {line}")
PY
