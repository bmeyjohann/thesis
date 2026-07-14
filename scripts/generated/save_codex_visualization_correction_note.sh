#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis
mkdir -p codex
cat > codex/unitree_height_scan_visualization_correction_20260708.md <<'EOF'
# Unitree height scan visualization correction

The previous height-scan rotation diagnostic selected a mapping by a weak aggregate distance score and reported it as fixed without visually validating every snapshot. This was wrong: the overlays still visibly failed to align.

Correct approach:
- Inspect generated images before claiming coordinate-frame fixes are solved.
- Prefer live sensor/world tensors over guessed grid conventions.
- For Unitree `terrain_scan`, use `terrain_scan.data.hit_pos_w` for visualization and derive any row/column-to-local mapping from those actual ray hit positions.
- Treat any score-only mapping as a hypothesis until checked against actual overlays across multiple snapshots.
EOF

python - <<'PY'
from pathlib import Path
p = Path("AGENTS.md")
note = "- Correction noted: before claiming a visualization/coordinate-frame fix is correct, inspect the generated image itself or derive the mapping from live sensor/world tensors. Do not select a scan-grid rotation from a weak aggregate score and present it as fixed when the overlays visibly do not align across snapshots.\n"
if p.exists():
    s = p.read_text()
    if note.strip() not in s:
        p.write_text(s.rstrip() + "\n" + note)
        print("updated AGENTS.md")
    else:
        print("AGENTS.md already contains note")
else:
    print("AGENTS.md not found")
print("saved codex/unitree_height_scan_visualization_correction_20260708.md")
PY
