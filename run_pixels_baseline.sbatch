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
TOTAL_STEPS=${TOTAL_STEPS:-1000000}
NUM_ENVS=${NUM_ENVS:-16}
BUFFER_SIZE=${BUFFER_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-4096}
LEARNING_STARTS=${LEARNING_STARTS:-2048}
PIXEL_WIDTH=${PIXEL_WIDTH:-64}
PIXEL_HEIGHT=${PIXEL_HEIGHT:-64}
DENSE_REWARD_SCALE=${DENSE_REWARD_SCALE:-0.05}
STEP_PENALTY=${STEP_PENALTY:-0.0}
LOG_INTERVAL=${LOG_INTERVAL:-2000}
SAVE_INTERVAL=${SAVE_INTERVAL:-100000}
PIXEL_FINAL_POOL=${PIXEL_FINAL_POOL:-8}
CNN_TINY=${CNN_TINY:-12,24,24}
KERNEL_TINY=${KERNEL_TINY:-5,5,3}
STRIDE_TINY=${STRIDE_TINY:-1,2,2}

CNN_SMALL=${CNN_SMALL:-16,32,32}
KERNEL_SMALL=${KERNEL_SMALL:-5,5,3}
STRIDE_SMALL=${STRIDE_SMALL:-1,2,2}

CNN_MEDIUM=${CNN_MEDIUM:-24,48,48}
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
--use_wandb --viz_on_checkpoint --viz_grid_resolution 32 --viz_quiver_stride 2"

echo "Launching baseline visual pixels sweep on ${ENV_ID}"

# Tiny CNN (minimal capacity)
CUDA_VISIBLE_DEVICES=0 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --pixel_conv_channels ${CNN_TINY} \
  --pixel_kernel_sizes ${KERNEL_TINY} \
  --pixel_strides ${STRIDE_TINY} \
  --pixel_final_pool ${PIXEL_FINAL_POOL} \
  --exp_name "pixels_dense_tiny_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_pixels_dense_tiny_${SLURM_JOB_ID}.log &
PID_A=$!

# Small CNN
CUDA_VISIBLE_DEVICES=1 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --pixel_conv_channels ${CNN_SMALL} \
  --pixel_kernel_sizes ${KERNEL_SMALL} \
  --pixel_strides ${STRIDE_SMALL} \
  --pixel_final_pool ${PIXEL_FINAL_POOL} \
  --exp_name "pixels_dense_small_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_pixels_dense_small_${SLURM_JOB_ID}.log &
PID_B=$!

# Medium CNN
CUDA_VISIBLE_DEVICES=2 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --pixel_conv_channels ${CNN_MEDIUM} \
  --pixel_kernel_sizes ${KERNEL_MEDIUM} \
  --pixel_strides ${STRIDE_MEDIUM} \
  --pixel_final_pool ${PIXEL_FINAL_POOL} \
  --exp_name "pixels_dense_medium_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_pixels_dense_medium_${SLURM_JOB_ID}.log &
PID_C=$!

# Large CNN (default)
CUDA_VISIBLE_DEVICES=3 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --pixel_conv_channels ${CNN_LARGE} \
  --pixel_kernel_sizes ${KERNEL_LARGE} \
  --pixel_strides ${STRIDE_LARGE} \
  --pixel_final_pool ${PIXEL_FINAL_POOL} \
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
