#!/bin/bash
#SBATCH --job-name=pixels_sweep_ref
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/pixels_sweep_ref_%j.out
#SBATCH --error=logs/pixels_sweep_ref_%j.err

set -euo pipefail

mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

# Base knobs (override via sbatch VAR=...)
ENV_ID=${ENV_ID:-visual-pointmaze-arena-v0}
TOTAL_STEPS=${TOTAL_STEPS:-600000}
NUM_ENVS=${NUM_ENVS:-16}
BUFFER_SIZE=${BUFFER_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-4096}
LEARNING_STARTS=${LEARNING_STARTS:-2048}
CRITIC_LR=${CRITIC_LR:-1e-4}
ACTOR_LR=${ACTOR_LR:-3e-4}
TAU=${TAU:-0.002}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
ALPHA_MIN=${ALPHA_MIN:-0.05}
PIXEL_FINAL_POOL=${PIXEL_FINAL_POOL:-8}

# Reward scales per run
SCALE_A=${SCALE_A:-0.2}
SCALE_B=${SCALE_B:-0.5}
SCALE_C=${SCALE_C:-0.5}
SCALE_D=${SCALE_D:-1.0}

# CNN specs per run
CNN_A=${CNN_A:-12,24,24}
KER_A=${KER_A:-5,5,3}
STR_A=${STR_A:-1,2,2}

# Single‑conv baseline (B)
CNN_B=${CNN_B:-16}
KER_B=${KER_B:-5}
STR_B=${STR_B:-1}

CNN_C=${CNN_C:-16,32,32}
KER_C=${KER_C:-5,5,3}
STR_C=${STR_C:-1,2,2}

CNN_D=${CNN_D:-32,64,64}
KER_D=${KER_D:-8,4,3}
STR_D=${STR_D:-4,2,1}

BASE_ARGS="--env_name ${ENV_ID} --obs_mode pixels --num_envs ${NUM_ENVS} \
--total_timesteps ${TOTAL_STEPS} --include_goal \
--reward_type dense --step_penalty 0.0 --buffer_size ${BUFFER_SIZE} \
--batch_size ${BATCH_SIZE} --learning_starts ${LEARNING_STARTS} \
--critic_learning_rate ${CRITIC_LR} --actor_learning_rate ${ACTOR_LR} \
--tau ${TAU} --max_grad_norm ${MAX_GRAD_NORM} --alpha_min ${ALPHA_MIN} \
--pixel_width 64 --pixel_height 64 --pixel_final_pool ${PIXEL_FINAL_POOL} \
--viz_on_checkpoint --viz_first_step 200000 --save_interval 100000 \
--use_wandb --viz_grid_resolution 32 --viz_quiver_stride 2"

echo "Launching pixel sweep (reference-stabilized) on ${ENV_ID}"

# A: tiny CNN, dense scale 0.2
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --dense_reward_scale ${SCALE_A} \
  --pixel_conv_channels ${CNN_A} --pixel_kernel_sizes ${KER_A} --pixel_strides ${STR_A} \
  --exp_name "pix_ref_A_tiny_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_pix_ref_A_tiny_${SLURM_JOB_ID}.log &
PID_A=$!

# B: single‑conv baseline, dense scale 0.5
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --dense_reward_scale ${SCALE_B} \
  --pixel_conv_channels ${CNN_B} --pixel_kernel_sizes ${KER_B} --pixel_strides ${STR_B} \
  --pixel_final_pool 0 \
  --exp_name "pix_ref_B_singleconv_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_pix_ref_B_singleconv_${SLURM_JOB_ID}.log &
PID_B=$!

# C: medium CNN, dense scale 0.5
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=2 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --dense_reward_scale ${SCALE_C} \
  --pixel_conv_channels ${CNN_C} --pixel_kernel_sizes ${KER_C} --pixel_strides ${STR_C} \
  --exp_name "pix_ref_C_medium_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_pix_ref_C_medium_${SLURM_JOB_ID}.log &
PID_C=$!

# D: large CNN, dense scale 1.0
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=3 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --dense_reward_scale ${SCALE_D} \
  --pixel_conv_channels ${CNN_D} --pixel_kernel_sizes ${KER_D} --pixel_strides ${STR_D} \
  --exp_name "pix_ref_D_large_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_pix_ref_D_large_${SLURM_JOB_ID}.log &
PID_D=$!

echo "PIDs: $PID_A $PID_B $PID_C $PID_D"

wait $PID_A; EA=$?
wait $PID_B; EB=$?
wait $PID_C; EC=$?
wait $PID_D; ED=$?

if [ $EA -ne 0 ] || [ $EB -ne 0 ] || [ $EC -ne 0 ] || [ $ED -ne 0 ]; then
  echo "One or more pixel sweep runs failed: $EA $EB $EC $ED"; exit 1
fi

echo "Pixel sweep (reference-stabilized) completed."

