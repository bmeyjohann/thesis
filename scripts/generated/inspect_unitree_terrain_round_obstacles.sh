#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis
echo "## matching terrain/obstacle definitions"
rg -n "discrete_obstacles|obstacle_width|terrain_generator|TerrainGenerator|height_scan|sphere|cylinder|box|cuboid|MeshCylinder|MeshBox" external/unitree_rl_mjlab -S | head -n 260
echo
echo "## relevant files"
find external/unitree_rl_mjlab -type f \( -name '*terrain*' -o -name '*navigation*' -o -name '*obstacle*' -o -name '*.py' \) | sort | head -n 260
