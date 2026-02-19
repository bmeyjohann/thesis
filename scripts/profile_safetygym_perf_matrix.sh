#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV=${CONDA_ENV:-fasttd3}
ENV_NAME=${ENV_NAME:-SafetyCarGoal2-v0}
TIMESTEPS=${TIMESTEPS:-512}
LEARNING_STARTS=${LEARNING_STARTS:-1000000}
BATCH_SIZE=${BATCH_SIZE:-64}
LOG_INTERVAL=${LOG_INTERVAL:-128}
SURFACE_MODE=${SURFACE_MODE:-default}
CAR_WHEEL_COMMAND_LIMIT=${CAR_WHEEL_COMMAND_LIMIT:-2.0}
CAR_FORCE_SCALE=${CAR_FORCE_SCALE:-2.0}
ENABLE_INTERVENTION_MATRIX=${ENABLE_INTERVENTION_MATRIX:-0}

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  HAS_DISPLAY=0
else
  HAS_DISPLAY=1
fi

mkdir -p logs/safetygym_perf
STAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/safetygym_perf/perf_matrix_${STAMP}.log"

RENDER_MODES=(none)
if [[ "${HAS_DISPLAY}" == "1" ]]; then
  RENDER_MODES+=(human)
fi

INTERVENTION_MODES=(0)
if [[ "${ENABLE_INTERVENTION_MATRIX}" == "1" && "${HAS_DISPLAY}" == "1" ]]; then
  INTERVENTION_MODES+=(1)
fi

UNCERTAINTY_MODES=(0 1)
UPDATE_MODES=(off on)

echo "Writing matrix outputs to ${OUTFILE}"
for render_mode in "${RENDER_MODES[@]}"; do
  for use_intervention in "${INTERVENTION_MODES[@]}"; do
    for uncertainty_on in "${UNCERTAINTY_MODES[@]}"; do
      for update_mode in "${UPDATE_MODES[@]}"; do
        cmd=(
          conda run -n "${CONDA_ENV}" python train_fast_sac_safetygym.py
          --env_name "${ENV_NAME}"
          --render_mode "${render_mode}"
          --surface_mode "${SURFACE_MODE}"
          --car_wheel_command_limit "${CAR_WHEEL_COMMAND_LIMIT}"
          --car_force_scale "${CAR_FORCE_SCALE}"
          --total_timesteps "${TIMESTEPS}"
          --learning_starts "${LEARNING_STARTS}"
          --batch_size "${BATCH_SIZE}"
          --log_interval "${LOG_INTERVAL}"
          --eval_interval 0
          --save_interval 0
          --reward_mode sparse
          --profile_timing
          --no_uncertainty_log_every_step
        )
        if [[ "${use_intervention}" == "1" ]]; then
          cmd+=(--use_intervention --controller_overlay_hz 10 --controller_fps_limit 30)
        fi
        if [[ "${uncertainty_on}" == "1" ]]; then
          cmd+=(--uncertainty_log_every_step)
        fi
        if [[ "${update_mode}" == "on" ]]; then
          cmd+=(--learning_starts 8)
        fi

        {
          echo "==== CASE render=${render_mode} intervention=${use_intervention} uncertainty=${uncertainty_on} updates=${update_mode} ===="
          printf '%q ' "${cmd[@]}"
          echo
          "${cmd[@]}"
          echo
        } | tee -a "${OUTFILE}"
      done
    done
  done
done

echo "Done. Review ${OUTFILE} for train_timing/* summaries."
