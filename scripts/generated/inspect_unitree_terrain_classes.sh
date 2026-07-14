#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis

python - <<'PY'
import inspect

import mjlab.terrains as terrain_gen

names = sorted(n for n in dir(terrain_gen) if "TerrainCfg" in n or "Obstacle" in n or "obstacle" in n.lower())
print("terrain names:")
for n in names:
    print(" ", n)

for n in names:
    if "Obstacle" not in n and "obstacle" not in n.lower():
        continue
    obj = getattr(terrain_gen, n)
    print(f"\n===== {n} =====")
    try:
        print(inspect.signature(obj))
    except Exception as exc:
        print("signature error:", exc)
    try:
        src = inspect.getsource(obj)
        print("\n".join(src.splitlines()[:120]))
    except Exception as exc:
        print("source error:", exc)
PY
