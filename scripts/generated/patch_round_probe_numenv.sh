#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis
python - <<'PY'
from pathlib import Path
p = Path("scripts/generated/run_unitree_round_obstacle_teacher_probe.sh")
s = p.read_text()
s = s.replace('"--num-envs", "16",\n    "--num-episodes", "20",', '"--num-envs", "1",\n    "--num-episodes", "10",')
p.write_text(s)
print("patched", p)
PY
