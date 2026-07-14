#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis
python - <<'PY'
from pathlib import Path
p = Path("scripts/generated/plot_unitree_multi_trajectories_probe.sh")
s = p.read_text()
old = "        if bool(done[0].detach().cpu().item()):\n            break\n"
new = "        done_tensor = torch.as_tensor(done)\n        if bool(done_tensor.reshape(-1)[0].detach().cpu().item()):\n            break\n"
if old not in s:
    raise SystemExit("target snippet not found")
p.write_text(s.replace(old, new))
print("patched scalar done handling")
PY
