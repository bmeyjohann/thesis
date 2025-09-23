#!/bin/bash
#SBATCH --job-name=teacher_student_sac
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/teacher_student_sac_%j.out
#SBATCH --error=logs/teacher_student_sac_%j.err

set -e
mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

ENV_ID=${ENV_ID:-pointmaze-arena-danger-lethal-v0}
TOTAL_STEPS=${TOTAL_STEPS:-2000000}
NUM_ENVS=${NUM_ENVS:-64}
TOL_A=${TOL_A:-30}
TOL_B=${TOL_B:-90}
HALF_STEPS_GLOBAL=$((TOTAL_STEPS/2))
ENABLE_AFTER_STEPS_PER_ENV=$((HALF_STEPS_GLOBAL/NUM_ENVS))
BUFFER_SIZE_DEFAULT=${BUFFER_SIZE:-51200}
BUFFER_SIZE_HALF=$((BUFFER_SIZE_DEFAULT/2))
DENIED_PENALTY=${DENIED_PENALTY:--1}
SAVE_INTERVAL=${SAVE_INTERVAL:-200000}
LOG_INTERVAL=${LOG_INTERVAL:-20000}
DENSE_REWARD_SCALE=${DENSE_REWARD_SCALE:-0.05}
STEP_PENALTY=${STEP_PENALTY:-0.001}

BASE_ARGS="--env_name ${ENV_ID} --num_envs ${NUM_ENVS} --total_timesteps ${TOTAL_STEPS} --include_goal --save_interval ${SAVE_INTERVAL} --log_interval ${LOG_INTERVAL} --use_wandb --dense_reward_scale ${DENSE_REWARD_SCALE} --step_penalty ${STEP_PENALTY}"

echo "Launching 6 FastSAC experiments on ${ENV_ID} with ${NUM_ENVS} envs each (concurrent)"

# 1) Teacher always on (tol A) with student penalty shaping (folded) and no lethal block
CUDA_VISIBLE_DEVICES=0 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type sparse \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --no_hard_block_lethal \
  --intervention_reward_mode penalty_student --intervention_reward_value 0.1 \
  --exp_name "teacher_always_tol${TOL_A}_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_teacher_always_tol${TOL_A}_${SLURM_JOB_ID}.log &
PID1=$!

# 2) Teacher always on (tol A) with bonus to teacher on intervention
CUDA_VISIBLE_DEVICES=1 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type sparse \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --no_hard_block_lethal \
  --intervention_reward_mode bonus_teacher --intervention_reward_value 0.1 \
  --exp_name "teacher_always_tol${TOL_A}_bonus_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_teacher_always_tol${TOL_A}_bonus_${SLURM_JOB_ID}.log &
PID2=$!

# 3) Teacher always on (tol A) but store student action and penalize it
CUDA_VISIBLE_DEVICES=2 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type sparse \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --no_hard_block_lethal \
  --intervention_reward_mode penalty_student --intervention_reward_value 0.1 \
  --exp_name "teacher_always_tol${TOL_A}_student_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_teacher_always_tol${TOL_A}_student_${SLURM_JOB_ID}.log &
PID3=$!

# 4) Student-only dense control
CUDA_VISIBLE_DEVICES=3 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type dense \
  --exp_name "student_only_dense_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_student_only_dense_${SLURM_JOB_ID}.log &
PID4=$!

echo "Waiting for first 4 experiments to finish..."
wait $PID1; E1=$?
wait $PID2; E2=$?
wait $PID3; E3=$?
wait $PID4; E4=$?

if [ $E1 -ne 0 ] || [ $E2 -ne 0 ] || [ $E3 -ne 0 ] || [ $E4 -ne 0 ]; then
  echo "One or more base experiments failed: $E1 $E2 $E3 $E4"; exit 1
fi

echo "Launching remaining 2 FastSAC variants (batch 2)"

# 5) Curriculum (tol A) with student penalty shaping
CUDA_VISIBLE_DEVICES=0 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type dense --reward_switch_after_steps ${HALF_STEPS_GLOBAL} \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --no_hard_block_lethal \
  --intervention_reward_mode penalty_student --intervention_reward_value 0.1 \
  --intervention_enable_after_steps ${ENABLE_AFTER_STEPS_PER_ENV} \
  --exp_name "curriculum_teacher_tol${TOL_A}_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_curriculum_teacher_tol${TOL_A}_${SLURM_JOB_ID}.log &
PID5=$!

# 6) Curriculum (tol A) with student_override shaping
CUDA_VISIBLE_DEVICES=1 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type dense --reward_switch_after_steps ${HALF_STEPS_GLOBAL} \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --no_hard_block_lethal \
  --intervention_reward_mode penalty_student --intervention_reward_value 0.1 \
  --intervention_enable_after_steps ${ENABLE_AFTER_STEPS_PER_ENV} \
  --exp_name "curriculum_teacher_tol${TOL_A}_student_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_curriculum_teacher_tol${TOL_A}_student_${SLURM_JOB_ID}.log &
PID6=$!

echo "Batch 2 PIDs: $PID5 $PID6"
wait $PID5; E5=$?
wait $PID6; E6=$?

if [ $E5 -ne 0 ] || [ $E6 -ne 0 ]; then
  echo "One or more batch-2 experiments failed: $E5 $E6"; exit 1
fi

echo "All FastSAC teacher–student experiments completed."
