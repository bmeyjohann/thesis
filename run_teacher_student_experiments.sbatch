#!/bin/bash
#SBATCH --job-name=teacher_student
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/teacher_student_%j.out
#SBATCH --error=logs/teacher_student_%j.err

# Teacher–Student experiments:
# 1) Teacher always on (angle tolerance A)
# 2) Teacher always on (angle tolerance B)
# 3) Curriculum: student-only dense first half, then teacher on + sparse reward (tol A)
# 4) Curriculum: student-only dense first half, then teacher on + sparse reward (tol B)

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

BASE_ARGS="--num_envs ${NUM_ENVS} \
--total_timesteps ${TOTAL_STEPS} \
--include_goal \
--config config/ogbench_config.yaml \
--env_name ${ENV_ID}"

echo "Launching 4 teacher-student experiments on ${ENV_ID} with ${NUM_ENVS} envs each"

# 1) Teacher always on (sparse, tol A)
CUDA_VISIBLE_DEVICES=0 python train_rsl_rl_integrated.py ${BASE_ARGS} \
  --reward_type sparse \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --hard_block_lethal \
  --experiment_name "${ENV_ID}_teacher_always_tol${TOL_A}_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_teacher_always_tol${TOL_A}_${SLURM_JOB_ID}.log &
PID1=$!

# 2) Teacher always on (sparse, tol B)
CUDA_VISIBLE_DEVICES=1 python train_rsl_rl_integrated.py ${BASE_ARGS} \
  --reward_type sparse \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_B} \
  --hard_block_lethal \
  --experiment_name "${ENV_ID}_teacher_always_tol${TOL_B}_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_teacher_always_tol${TOL_B}_${SLURM_JOB_ID}.log &
PID2=$!

# 3) Curriculum: dense first half, teacher on + sparse afterward (tol A)
CUDA_VISIBLE_DEVICES=2 python train_rsl_rl_integrated.py ${BASE_ARGS} \
  --reward_type dense \
  --dense_reward_scale 0.01 \
  --reward_switch_after_steps ${HALF_STEPS_GLOBAL} \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --hard_block_lethal \
  --intervention_enable_after_steps ${ENABLE_AFTER_STEPS_PER_ENV} \
  --experiment_name "${ENV_ID}_curriculum_teacher_tol${TOL_A}_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_curriculum_teacher_tol${TOL_A}_${SLURM_JOB_ID}.log &
PID3=$!

# 4) Curriculum: dense first half, teacher on + sparse afterward (tol B)
CUDA_VISIBLE_DEVICES=3 python train_rsl_rl_integrated.py ${BASE_ARGS} \
  --reward_type dense \
  --dense_reward_scale 0.01 \
  --reward_switch_after_steps ${HALF_STEPS_GLOBAL} \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_B} \
  --hard_block_lethal \
  --intervention_enable_after_steps ${ENABLE_AFTER_STEPS_PER_ENV} \
  --experiment_name "${ENV_ID}_curriculum_teacher_tol${TOL_B}_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_curriculum_teacher_tol${TOL_B}_${SLURM_JOB_ID}.log &
PID4=$!

echo "PIDs: $PID1 $PID2 $PID3 $PID4"
wait $PID1; E1=$?
wait $PID2; E2=$?
wait $PID3; E3=$?
wait $PID4; E4=$?

if [ $E1 -ne 0 ] || [ $E2 -ne 0 ] || [ $E3 -ne 0 ] || [ $E4 -ne 0 ]; then
  echo "One or more experiments failed: $E1 $E2 $E3 $E4"; exit 1
fi

echo "All teacher–student experiments completed."

