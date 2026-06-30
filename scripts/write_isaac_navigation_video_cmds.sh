#!/usr/bin/env bash
set -euo pipefail

WIN_REPO="${WIN_REPO:-/mnt/c/Data/thesis}"
RUN_ROOT="${WIN_REPO}/logs/isaac_navigation_scripted"
CONDA_ACTIVATE='C:\Users\benja\miniconda3\Scripts\activate.bat'
CONDA_ENV='C:\Users\benja\miniconda3\envs\env_isaaclab'
POLICY='C:\Data\thesis\logs\rsl_rl\unitree_go2_flat\2025-11-23_13-46-53_official_task_ppo_12893598\exported\policy.pt'
EVAL_SCRIPT='C:\Data\thesis\safe-locomotion\scripts\rsl_rl\eval_navigation_scripted.py'
ISAACLAB='C:\Data\thesis\IsaacLab\isaaclab.bat'

write_cmd() {
  local run_name="$1"
  local task="$2"
  local controller="$3"
  local win_out="C:\\Data\\thesis\\logs\\isaac_navigation_scripted\\${run_name}"
  local out_dir="${RUN_ROOT}/${run_name}"
  mkdir -p "${out_dir}"
  cat > "${out_dir}/run_eval_video.cmd" <<EOF
@echo on
set NAV_LOW_LEVEL_CFG=official
set MPLCONFIGDIR=%TEMP%\\mplconfig_isaac_nav
call "${CONDA_ACTIVATE}" "${CONDA_ENV}"
if errorlevel 1 exit /b %errorlevel%
cd /d "C:\\Data\\thesis"
if errorlevel 1 exit /b %errorlevel%
"${ISAACLAB}" -p "${EVAL_SCRIPT}" --headless --video --video_length 350 --video_camera_mode scene --device cpu --task "${task}" --low_level_policy_path "${POLICY}" --num_envs 1 --num_episodes 3 --max_steps 500 --output_dir "${win_out}" --seed 0 --controller "${controller}" --lin_gain 0.35 --yaw_gain 0.8 --max_lin_vel 0.35 --max_yaw_vel 0.6 --forward_heading_deadband 0.35 --teacher_obstacle_x 1.0 --teacher_obstacle_y 0.0 --teacher_obstacle_radius 0.65 --teacher_obstacle_margin 0.45 --teacher_obstacle_side left --stop_on_success
exit /b %errorlevel%
EOF
  unix2dos "${out_dir}/run_eval_video.cmd" >/dev/null 2>&1 || true
  echo "${out_dir}/run_eval_video.cmd"
}

write_cmd "win_straightline_official_isaac_video_probe" "Isaac-Navigation-NoObstacles-Flat-Go2-v0" "goal_straight"
write_cmd "win_single_obstacle_teacher_isaac_video_probe" "Isaac-Navigation-SingleObstacle-Flat-Go2-v0" "obstacle_teacher"
