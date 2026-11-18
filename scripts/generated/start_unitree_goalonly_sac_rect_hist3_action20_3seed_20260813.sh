#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis
mkdir -p logs/unitree_mjlab/background
log="logs/unitree_mjlab/background/goalonly_sac_rect_hist3_action20_3seed_$(date +%Y%m%d_%H%M%S).log"
nohup ./scripts/generated/run_unitree_goalonly_sac_rect_hist3_action20_3seed_20260813.sh >"$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > logs/unitree_mjlab/background/goalonly_sac_rect_hist3_action20_3seed.pid
printf 'PID=%s\nLOG=%s\n' "$pid" "$log"
