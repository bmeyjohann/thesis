#!/bin/bash
#SBATCH --job-name=ts_sac_batchB_cf
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/ts_sac_batchB_cf_%j.out
#SBATCH --error=logs/ts_sac_batchB_cf_%j.err

set -e
mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

ENV_ID=${ENV_ID:-pointmaze-arena-danger-lethal-v0}
TOTAL_STEPS=${TOTAL_STEPS:-2000000}
NUM_ENVS=${NUM_ENVS:-64}
TOL_A=${TOL_A:-30}
HALF_STEPS_GLOBAL=$((TOTAL_STEPS/2))
ENABLE_AFTER_STEPS_PER_ENV=$((HALF_STEPS_GLOBAL/NUM_ENVS))
SAVE_INTERVAL=${SAVE_INTERVAL:-200000}
LOG_INTERVAL=${LOG_INTERVAL:-20000}
DENSE_REWARD_SCALE=${DENSE_REWARD_SCALE:-0.05}
STEP_PENALTY=${STEP_PENALTY:-0.001}

BASE_ARGS="--env_name ${ENV_ID} --num_envs ${NUM_ENVS} --total_timesteps ${TOTAL_STEPS} --include_goal \
--save_interval ${SAVE_INTERVAL} --log_interval ${LOG_INTERVAL} --use_wandb \
--dense_reward_scale ${DENSE_REWARD_SCALE} --step_penalty ${STEP_PENALTY} \
--cf_buffer_enable --cf_sample_ratio 0.5 --cf_q_weight 1.0"

echo "Launching 4 FastSAC CF experiments (Batch B) on ${ENV_ID}"

# 1) Teacher always on (tol A) with CF penalty -1.0 (no shaping)
CUDA_VISIBLE_DEVICES=0 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type sparse \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --no_hard_block_lethal \
  --intervention_reward_mode none --cf_penalty 1.0 \
  --exp_name "teacher_always_tol${TOL_A}_cfm1_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_teacher_always_tol${TOL_A}_cfm1_${SLURM_JOB_ID}.log &
PID1=$!

# 2) Teacher always on (tol A) with CF penalty 0.0 (control)
CUDA_VISIBLE_DEVICES=1 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type sparse \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --no_hard_block_lethal \
  --intervention_reward_mode none --cf_penalty 0.0 \
  --exp_name "teacher_always_tol${TOL_A}_cf0_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_teacher_always_tol${TOL_A}_cf0_${SLURM_JOB_ID}.log &
PID2=$!

# 3) Curriculum (tol A), CF penalty -1.0 + reset RB on switch
CUDA_VISIBLE_DEVICES=2 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type dense --reward_switch_after_steps ${HALF_STEPS_GLOBAL} \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --no_hard_block_lethal \
  --intervention_reward_mode none --cf_penalty 1.0 --reset_replay_on_switch \
  --intervention_enable_after_steps ${ENABLE_AFTER_STEPS_PER_ENV} \
  --exp_name "curriculum_teacher_tol${TOL_A}_cfm1_reset_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_curriculum_teacher_tol${TOL_A}_cfm1_reset_${SLURM_JOB_ID}.log &
PID3=$!

# 4) Curriculum (tol A), CF penalty -1.0 (no reset)
CUDA_VISIBLE_DEVICES=3 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type dense --reward_switch_after_steps ${HALF_STEPS_GLOBAL} \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --no_hard_block_lethal \
  --intervention_reward_mode none --cf_penalty 1.0 \
  --intervention_enable_after_steps ${ENABLE_AFTER_STEPS_PER_ENV} \
  --exp_name "curriculum_teacher_tol${TOL_A}_cfm1_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_curriculum_teacher_tol${TOL_A}_cfm1_${SLURM_JOB_ID}.log &
PID4=$!

echo "PIDs: $PID1 $PID2 $PID3 $PID4"
wait $PID1; E1=$?
wait $PID2; E2=$?
wait $PID3; E3=$?
wait $PID4; E4=$?

if [ $E1 -ne 0 ] || [ $E2 -ne 0 ] || [ $E3 -ne 0 ] || [ $E4 -ne 0 ]; then
  echo "One or more experiments failed: $E1 $E2 $E3 $E4"; exit 1
fi

echo "Batch B (CF) completed."
