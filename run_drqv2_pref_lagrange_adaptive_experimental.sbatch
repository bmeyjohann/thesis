#!/bin/bash
#SBATCH --job-name=drqv2_pref_lagrange_adapt_exp
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/drqv2_pref_lagrange_adapt_exp_%j.out
#SBATCH --error=logs/drqv2_pref_lagrange_adapt_exp_%j.err

set -euo pipefail
mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-${SLURM_JOB_ID}}
mkdir -p "${MPLCONFIGDIR}"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${SLURM_TMPDIR:-/tmp}/xdg_cache_${SLURM_JOB_ID}}
mkdir -p "${XDG_CACHE_HOME}"
export WANDB_DIR=${WANDB_DIR:-${SLURM_TMPDIR:-/tmp}/wandb_${SLURM_JOB_ID}}
mkdir -p "${WANDB_DIR}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

ENV_ID=${ENV_ID:-pointmaze-arena-danger-wall-v0}
TOTAL_STEPS=${TOTAL_STEPS:-1500000}
SAVE_INTERVAL=${SAVE_INTERVAL:-200000}
LOG_INTERVAL=${LOG_INTERVAL:-1000}
PREF_MARGIN=${PREF_MARGIN:-0.1}
PREF_WEIGHT=${PREF_WEIGHT:-1.0}
REPLAY_WORKERS=${REPLAY_WORKERS:-0}

LAMBDA_INIT=${LAMBDA_INIT:-0.0}
LAMBDA_LR=${LAMBDA_LR:-1e-3}
LAMBDA_MAX=${LAMBDA_MAX:-10.0}
LAMBDA_EMA=${LAMBDA_EMA:-0.9}
VIOLATION_CLIP=${VIOLATION_CLIP:-10.0}

BASE_ARGS="--env_name ${ENV_ID} --total_timesteps ${TOTAL_STEPS} --include_goal \
--save_interval ${SAVE_INTERVAL} --log_interval ${LOG_INTERVAL} --use_wandb \
--reward_type dense --dense_reward_scale 0.05 --step_penalty 0.002 \
--pref_buffer_enable --pref_capacity 100000 --pref_chunk_size 64 \
--pref_sample_ratio 0.3 --pref_rank_margin ${PREF_MARGIN} --pref_rank_weight ${PREF_WEIGHT} \
--use_intervention --intervention_mode agent --teacher_type bfs \
--pixel_width 84 --pixel_height 84 --frame_stack 3 --action_repeat 2 \
--replay_buffer_num_workers ${REPLAY_WORKERS} \
--disagreement_hist_edges 0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0,1.5,2.0 \
--eval_every_frames 200000 --num_eval_episodes 3 --viz_on_checkpoint --viz_grid_resolution 64 \
--viz_quiver_stride 2 --viz_device cpu --viz_seed 0"

echo "Launching DrQ-v2 pref baseline vs lagrangian and adaptive UTD on ${ENV_ID}"

CUDA_VISIBLE_DEVICES=0 python -u train_drqv2_ogbench.py ${BASE_ARGS} \
  --pref_loss_type margin \
  --exp_name "drq_pref_margin_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_drq_pref_margin_${SLURM_JOB_ID}.log &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python -u train_drqv2_ogbench.py ${BASE_ARGS} \
  --pref_loss_type lagrangian \
  --pref_lambda_init ${LAMBDA_INIT} --pref_lambda_lr ${LAMBDA_LR} \
  --pref_lambda_max ${LAMBDA_MAX} --pref_lambda_ema ${LAMBDA_EMA} \
  --pref_violation_clip ${VIOLATION_CLIP} \
  --exp_name "drq_pref_lagrange_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_drq_pref_lagrange_${SLURM_JOB_ID}.log &
PID2=$!

CUDA_VISIBLE_DEVICES=2 python -u train_drqv2_ogbench.py ${BASE_ARGS} \
  --pref_loss_type margin \
  --adaptive_utd --adaptive_utd_max_updates 5 \
  --exp_name "drq_pref_adapt5_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_drq_pref_adapt5_${SLURM_JOB_ID}.log &
PID3=$!

CUDA_VISIBLE_DEVICES=3 python -u train_drqv2_ogbench.py ${BASE_ARGS} \
  --pref_loss_type margin \
  --adaptive_utd --adaptive_utd_max_updates 20 \
  --exp_name "drq_pref_adapt20_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_drq_pref_adapt20_${SLURM_JOB_ID}.log &
PID4=$!

echo "PIDs: $PID1 $PID2 $PID3 $PID4"
wait $PID1; E1=$?
wait $PID2; E2=$?
wait $PID3; E3=$?
wait $PID4; E4=$?

if [ $E1 -ne 0 ] || [ $E2 -ne 0 ] || [ $E3 -ne 0 ] || [ $E4 -ne 0 ]; then
  echo "One or more DrQ preference runs failed: $E1 $E2 $E3 $E4"
  exit 1
fi

echo "DrQ-v2 pref lagrangian/adaptive runs completed."
