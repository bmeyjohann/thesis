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

BASE_ARGS="--env_name ${ENV_ID} --num_envs ${NUM_ENVS} --total_timesteps ${TOTAL_STEPS} --include_goal --save_interval ${SAVE_INTERVAL} --log_interval ${LOG_INTERVAL}"

echo "Launching 4 FastSAC teacher-student experiments on ${ENV_ID} with ${NUM_ENVS} envs each"

# 1) Teacher always on (sparse, tol A)
CUDA_VISIBLE_DEVICES=0 python train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type sparse \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --hard_block_lethal \
  --exp_name "teacher_always_tol${TOL_A}_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_teacher_always_tol${TOL_A}_${SLURM_JOB_ID}.log &
PID1=$!

# 2) Teacher always on (sparse, tol B)
CUDA_VISIBLE_DEVICES=1 python train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type sparse \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_B} \
  --hard_block_lethal \
  --exp_name "teacher_always_tol${TOL_B}_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_teacher_always_tol${TOL_B}_${SLURM_JOB_ID}.log &
PID2=$!

# 3) Curriculum: dense first half, then teacher + sparse (tol A)
CUDA_VISIBLE_DEVICES=2 python train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type dense --dense_reward_scale 0.01 \
  --reward_switch_after_steps ${HALF_STEPS_GLOBAL} \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --hard_block_lethal \
  --intervention_enable_after_steps ${ENABLE_AFTER_STEPS_PER_ENV} \
  --exp_name "curriculum_teacher_tol${TOL_A}_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_curriculum_teacher_tol${TOL_A}_${SLURM_JOB_ID}.log &
PID3=$!

# 4) Curriculum: dense first half, then teacher + sparse (tol B)
CUDA_VISIBLE_DEVICES=3 python train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type dense --dense_reward_scale 0.01 \
  --reward_switch_after_steps ${HALF_STEPS_GLOBAL} \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_B} \
  --hard_block_lethal \
  --intervention_enable_after_steps ${ENABLE_AFTER_STEPS_PER_ENV} \
  --exp_name "curriculum_teacher_tol${TOL_B}_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_curriculum_teacher_tol${TOL_B}_${SLURM_JOB_ID}.log &
PID4=$!

echo "PIDs: $PID1 $PID2 $PID3 $PID4"
wait $PID1; E1=$?
wait $PID2; E2=$?
wait $PID3; E3=$?
wait $PID4; E4=$?

if [ $E1 -ne 0 ] || [ $E2 -ne 0 ] || [ $E3 -ne 0 ] || [ $E4 -ne 0 ]; then
  echo "One or more base experiments failed: $E1 $E2 $E3 $E4"; exit 1
fi

echo "Base FastSAC teacher–student experiments completed."

echo "Launching FastSAC variants (buffer_size=${BUFFER_SIZE_HALF}, denied_penalty=${DENIED_PENALTY})"

# 5) Teacher always on (tol A) with half-sized replay buffer
CUDA_VISIBLE_DEVICES=0 python train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type sparse \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --hard_block_lethal \
  --buffer_size ${BUFFER_SIZE_HALF} \
  --exp_name "teacher_always_tol${TOL_A}_bufferhalf_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_teacher_always_tol${TOL_A}_bufferhalf_${SLURM_JOB_ID}.log &
PID5=$!

# 6) Teacher always on (tol A) storing denied student actions with penalty
CUDA_VISIBLE_DEVICES=1 python train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type sparse \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --hard_block_lethal \
  --store_denied_actions --denied_action_penalty ${DENIED_PENALTY} \
  --exp_name "teacher_always_tol${TOL_A}_denied_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_teacher_always_tol${TOL_A}_denied_${SLURM_JOB_ID}.log &
PID6=$!

echo "Variant PIDs: $PID5 $PID6"
wait $PID5; E5=$?
wait $PID6; E6=$?

if [ $E5 -ne 0 ] || [ $E6 -ne 0 ]; then
  echo "One or more variant experiments failed: $E5 $E6"; exit 1
fi

echo "All FastSAC teacher–student experiments completed."
