#!/bin/bash
#SBATCH --job-name=simple_arena_sac
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/simple_arena_sac_%j.out
#SBATCH --error=logs/simple_arena_sac_%j.err

set -e
mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

# Simpler environment without lethal cells
ENV_ID=${ENV_ID:-pointmaze-arena-v0}
TOTAL_STEPS=${TOTAL_STEPS:-1000000}
NUM_ENVS=${NUM_ENVS:-64}
TOL=${TOL:-45}

# Reward shaping for student learning
DENSE_REWARD_SCALE=${DENSE_REWARD_SCALE:-0.05}
STEP_PENALTY=${STEP_PENALTY:-0.01}

SAVE_INTERVAL=${SAVE_INTERVAL:-100000}
LOG_INTERVAL=${LOG_INTERVAL:-20000}

BASE_ARGS="--env_name ${ENV_ID} --num_envs ${NUM_ENVS} --total_timesteps ${TOTAL_STEPS} \
--include_goal --dense_reward_scale ${DENSE_REWARD_SCALE} --step_penalty ${STEP_PENALTY} \
--save_interval ${SAVE_INTERVAL} --log_interval ${LOG_INTERVAL} --use_wandb"

echo "Launching 4 FastSAC simple-arena experiments (batch 1)"

# 1) Student-only dense (baseline learning sanity)
CUDA_VISIBLE_DEVICES=0 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type dense \
  --exp_name "simple_student_dense_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_simple_student_dense_${SLURM_JOB_ID}.log &
PID1=$!

# 2) Teacher tol=45 with denied-action penalty (sparse)
CUDA_VISIBLE_DEVICES=1 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type sparse \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL} \
  --no_hard_block_lethal \
  --store_denied_actions --denied_action_penalty -1.0 \
  --exp_name "simple_teacher_tol${TOL}_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_simple_teacher_tol${TOL}_${SLURM_JOB_ID}.log &
PID2=$!

# 3) Teacher tol=45 without penalty (sparse)
CUDA_VISIBLE_DEVICES=2 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type sparse \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL} \
  --no_hard_block_lethal \
  --exp_name "simple_teacher_tol${TOL}_nop_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_simple_teacher_tol${TOL}_nop_${SLURM_JOB_ID}.log &
PID3=$!

# 4) Curriculum: dense first half, then teacher+sparse (penalty on)
CUDA_VISIBLE_DEVICES=3 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type dense --reward_switch_after_steps $((TOTAL_STEPS/2)) \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL} \
  --no_hard_block_lethal \
  --store_denied_actions --denied_action_penalty -1.0 \
  --intervention_enable_after_steps $((TOTAL_STEPS/(2*NUM_ENVS))) \
  --exp_name "simple_curriculum_tol${TOL}_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_simple_curriculum_tol${TOL}_${SLURM_JOB_ID}.log &
PID4=$!

echo "Waiting for batch 1 PIDs: $PID1 $PID2 $PID3 $PID4"
wait $PID1; E1=$?
wait $PID2; E2=$?
wait $PID3; E3=$?
wait $PID4; E4=$?

if [ $E1 -ne 0 ] || [ $E2 -ne 0 ] || [ $E3 -ne 0 ] || [ $E4 -ne 0 ]; then
  echo "One or more batch-1 experiments failed: $E1 $E2 $E3 $E4"; exit 1
fi

echo "Launching 2 simple-arena variants (batch 2)"

# 5) Teacher tol=90 with penalty (sparse)
CUDA_VISIBLE_DEVICES=0 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type sparse \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value 90 \
  --no_hard_block_lethal \
  --store_denied_actions --denied_action_penalty -1.0 \
  --exp_name "simple_teacher_tol90_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_simple_teacher_tol90_${SLURM_JOB_ID}.log &
PID5=$!

# 6) Curriculum tol=45 no penalty (control)
CUDA_VISIBLE_DEVICES=1 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type dense --reward_switch_after_steps $((TOTAL_STEPS/2)) \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL} \
  --no_hard_block_lethal \
  --intervention_enable_after_steps $((TOTAL_STEPS/(2*NUM_ENVS))) \
  --exp_name "simple_curriculum_tol${TOL}_nop_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_simple_curriculum_tol${TOL}_nop_${SLURM_JOB_ID}.log &
PID6=$!

echo "Waiting for batch 2 PIDs: $PID5 $PID6"
wait $PID5; E5=$?
wait $PID6; E6=$?

if [ $E5 -ne 0 ] || [ $E6 -ne 0 ]; then
  echo "One or more batch-2 experiments failed: $E5 $E6"; exit 1
fi

echo "All simple-arena FastSAC experiments completed."
