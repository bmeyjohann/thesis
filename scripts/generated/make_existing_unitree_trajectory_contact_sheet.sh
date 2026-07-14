#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis
export MPLBACKEND=Agg

python - <<'PY'
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

paths = [
    Path("visualizations/unitree_nav_rollout_20260705_teacher_strict_plot_a/scan_teacher_topdown_rollout.png"),
    Path("visualizations/unitree_nav_rollout_20260705_teacher_audit_failure/scan_teacher_topdown_rollout.png"),
    Path("visualizations/unitree_nav_rollout_20260706_rolloutplanner_farclear/scan_teacher_topdown_rollout.png"),
    Path("visualizations/unitree_nav_debug_20260703_blocked_bypass045_strict_audit_plot/scan_teacher_topdown_rollout.png"),
    Path("visualizations/unitree_nav_layout_samples_20260705/big_obstacles_goal_through_obstacle_seed31_4.png"),
    Path("visualizations/unitree_nav_layout_samples_20260705/big_obstacles_clearance_seed21_4.png"),
]
existing = [p for p in paths if p.exists()]
if not existing:
    raise SystemExit("no existing Unitree trajectory/layout PNGs found")

out_dir = Path("visualizations/unitree_nav_multi_trajectories/existing_contact_sheet_20260707")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "existing_unitree_trajectory_contact_sheet.png"

cols = 3
rows = (len(existing) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.4 * rows), dpi=150, squeeze=False)
for idx, p in enumerate(existing):
    ax = axes[idx // cols][idx % cols]
    ax.imshow(mpimg.imread(p))
    ax.set_title(p.parent.name, fontsize=8)
    ax.axis("off")
for idx in range(len(existing), rows * cols):
    axes[idx // cols][idx % cols].axis("off")
fig.suptitle("Existing Unitree trajectory/layout plots: obstacles + collision markers where present", fontsize=12)
fig.tight_layout()
fig.savefig(out_path)
plt.close(fig)
print(out_path.resolve())
PY
