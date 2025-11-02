#!/bin/bash
#SBATCH --job-name=pixels_baseline
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/pixels_baseline_%j.out
#SBATCH --error=logs/pixels_baseline_%j.err

set -euo pipefail

mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

ENV_ID=${ENV_ID:-visual-pointmaze-arena-v0}
TOTAL_STEPS=${TOTAL_STEPS:-600000}
NUM_ENVS=${NUM_ENVS:-16}
BUFFER_SIZE=${BUFFER_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-4096}
LEARNING_STARTS=${LEARNING_STARTS:-2048}
PIXEL_WIDTH=${PIXEL_WIDTH:-64}
PIXEL_HEIGHT=${PIXEL_HEIGHT:-64}
DENSE_REWARD_SCALE=${DENSE_REWARD_SCALE:-1.0}
DENSE_REWARD_SCALE_TINY=${DENSE_REWARD_SCALE_TINY:-0.2}
DENSE_REWARD_SCALE_SMALL=${DENSE_REWARD_SCALE_SMALL:-0.5}
DENSE_REWARD_SCALE_MEDIUM=${DENSE_REWARD_SCALE_MEDIUM:-0.5}
DENSE_REWARD_SCALE_LARGE=${DENSE_REWARD_SCALE_LARGE:-1.0}
STEP_PENALTY=${STEP_PENALTY:-0.0}
LOG_INTERVAL=${LOG_INTERVAL:-2000}
SAVE_INTERVAL=${SAVE_INTERVAL:-100000}
PIXEL_FINAL_POOL=${PIXEL_FINAL_POOL:-8}
ALPHA_MIN=${ALPHA_MIN:-0.0}
ALPHA_MIN_SMALL=${ALPHA_MIN_SMALL:-0.05}
ALPHA_MIN_MEDIUM=${ALPHA_MIN_MEDIUM:-0.05}
ALPHA_MIN_LARGE=${ALPHA_MIN_LARGE:-0.05}
ALPHA_MAX=${ALPHA_MAX:-0.2}
ALPHA_FREEZE=${ALPHA_FREEZE:-50000}
CNN_TINY=${CNN_TINY:-12,24,24}
KERNEL_TINY=${KERNEL_TINY:-5,5,3}
STRIDE_TINY=${STRIDE_TINY:-1,2,2}

CNN_SMALL=${CNN_SMALL:-16}
KERNEL_SMALL=${KERNEL_SMALL:-5}
STRIDE_SMALL=${STRIDE_SMALL:-1}

CNN_MEDIUM=${CNN_MEDIUM:-16,32,32}
KERNEL_MEDIUM=${KERNEL_MEDIUM:-5,5,3}
STRIDE_MEDIUM=${STRIDE_MEDIUM:-1,2,2}

CNN_LARGE=${CNN_LARGE:-32,64,64}
KERNEL_LARGE=${KERNEL_LARGE:-8,4,3}
STRIDE_LARGE=${STRIDE_LARGE:-4,2,1}

BASE_ARGS="--env_name ${ENV_ID} --obs_mode pixels --num_envs ${NUM_ENVS} \
--total_timesteps ${TOTAL_STEPS} --include_goal \
--reward_type dense --dense_reward_scale ${DENSE_REWARD_SCALE} \
--step_penalty ${STEP_PENALTY} --buffer_size ${BUFFER_SIZE} \
--pixel_width ${PIXEL_WIDTH} --pixel_height ${PIXEL_HEIGHT} \
--pixel_final_pool ${PIXEL_FINAL_POOL} \
--batch_size ${BATCH_SIZE} --learning_starts ${LEARNING_STARTS} \
--log_interval ${LOG_INTERVAL} --save_interval ${SAVE_INTERVAL} \
--alpha_min ${ALPHA_MIN} --alpha_max ${ALPHA_MAX} --alpha_freeze_steps ${ALPHA_FREEZE} \
--viz_first_step 200000 \
--use_wandb --viz_on_checkpoint --viz_grid_resolution 32 --viz_quiver_stride 2"

echo "Launching baseline visual pixels sweep on ${ENV_ID}"

# Tiny CNN (minimal capacity)
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --pixel_conv_channels ${CNN_TINY} \
  --pixel_kernel_sizes ${KERNEL_TINY} \
  --pixel_strides ${STRIDE_TINY} \
  --pixel_final_pool ${PIXEL_FINAL_POOL} \
  --dense_reward_scale ${DENSE_REWARD_SCALE_TINY} \
  --exp_name "pixels_dense_tiny_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_pixels_dense_tiny_${SLURM_JOB_ID}.log &
PID_A=$!

# Small CNN (single-layer)
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --pixel_conv_channels ${CNN_LINEAR} \
  --pixel_kernel_sizes ${KERNEL_LINEAR} \
  --pixel_strides ${STRIDE_LINEAR} \
  --pixel_final_pool 0 \
  --alpha_min ${ALPHA_MIN_SMALL} \
  --dense_reward_scale ${DENSE_REWARD_SCALE_SMALL} \
  --exp_name "pixels_dense_small_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_pixels_dense_small_${SLURM_JOB_ID}.log &
PID_B=$!

# Medium CNN
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=2 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --pixel_conv_channels ${CNN_MEDIUM} \
  --pixel_kernel_sizes ${KERNEL_MEDIUM} \
  --pixel_strides ${STRIDE_MEDIUM} \
  --pixel_final_pool ${PIXEL_FINAL_POOL} \
  --alpha_min ${ALPHA_MIN_MEDIUM} \
  --dense_reward_scale ${DENSE_REWARD_SCALE_MEDIUM} \
  --exp_name "pixels_dense_medium_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_pixels_dense_medium_${SLURM_JOB_ID}.log &
PID_C=$!

# Large CNN (default)
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=3 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --pixel_conv_channels ${CNN_LARGE} \
  --pixel_kernel_sizes ${KERNEL_LARGE} \
  --pixel_strides ${STRIDE_LARGE} \
  --pixel_final_pool ${PIXEL_FINAL_POOL} \
  --alpha_min ${ALPHA_MIN_LARGE} \
  --dense_reward_scale ${DENSE_REWARD_SCALE_LARGE} \
  --exp_name "pixels_dense_large_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_pixels_dense_large_${SLURM_JOB_ID}.log &
PID_D=$!

echo "PIDs: $PID_A $PID_B $PID_C $PID_D"

wait $PID_A; EXIT_A=$?
wait $PID_B; EXIT_B=$?
wait $PID_C; EXIT_C=$?
wait $PID_D; EXIT_D=$?

if [ $EXIT_A -ne 0 ] || [ $EXIT_B -ne 0 ] || [ $EXIT_C -ne 0 ] || [ $EXIT_D -ne 0 ]; then
  echo "One or more baseline runs failed: $EXIT_A $EXIT_B $EXIT_C $EXIT_D"
  exit 1
fi

echo "Baseline pixel experiments completed."
CNN_LINEAR=${CNN_LINEAR:-16}
KERNEL_LINEAR=${KERNEL_LINEAR:-5}
STRIDE_LINEAR=${STRIDE_LINEAR:-1}
