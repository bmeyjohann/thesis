#!/bin/bash
#SBATCH --job-name=ts_sac_batchE_resetcrit
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/ts_sac_batchE_resetcrit_%j.out
#SBATCH --error=logs/ts_sac_batchE_resetcrit_%j.err

set -e
mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

ENV_ID=${ENV_ID:-pointmaze-arena-danger-lethal-v0}
TOTAL_STEPS=${TOTAL_STEPS:-3000000}
NUM_ENVS=${NUM_ENVS:-64}
TOL_A=${TOL_A:-30}
SWITCH_STEPS_GLOBAL=$((TOTAL_STEPS/3))
ENABLE_AFTER_STEPS_PER_ENV=$((SWITCH_STEPS_GLOBAL/NUM_ENVS))
SAVE_INTERVAL=${SAVE_INTERVAL:-200000}
LOG_INTERVAL=${LOG_INTERVAL:-20000}
DENSE_REWARD_SCALE=${DENSE_REWARD_SCALE:-0.05}
STEP_PENALTY=${STEP_PENALTY:-0.002}

BASE_ARGS="--env_name ${ENV_ID} --num_envs ${NUM_ENVS} --total_timesteps ${TOTAL_STEPS} --include_goal \
--save_interval ${SAVE_INTERVAL} --log_interval ${LOG_INTERVAL} --use_wandb \
--dense_reward_scale ${DENSE_REWARD_SCALE} --step_penalty ${STEP_PENALTY} \
--reset_replay_on_switch --reset_critic_on_switch"

echo "Launching FastSAC critic-reset experiments (Batch E) on ${ENV_ID}"

# 1) Terminal-student penalty=1.0, replay+critic reset
CUDA_VISIBLE_DEVICES=0 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type dense --reward_switch_after_steps ${SWITCH_STEPS_GLOBAL} \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --no_hard_block_lethal \
  --intervention_reward_mode penalty_student --intervention_reward_value 1.0 \
  --intervention_enable_after_steps ${ENABLE_AFTER_STEPS_PER_ENV} \
  --exp_name "E_term_pen1_resetcrit_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_E_term_pen1_resetcrit_${SLURM_JOB_ID}.log &
PID1=$!

# 2) Counterfactual buffer penalty=1.0, replay+critic reset
CUDA_VISIBLE_DEVICES=1 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type dense --reward_switch_after_steps ${SWITCH_STEPS_GLOBAL} \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --no_hard_block_lethal \
  --cf_buffer_enable --cf_sample_ratio 0.5 --cf_q_weight 1.0 --cf_penalty 1.0 \
  --intervention_reward_mode none \
  --intervention_enable_after_steps ${ENABLE_AFTER_STEPS_PER_ENV} \
  --exp_name "E_cf_pen1_resetcrit_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_E_cf_pen1_resetcrit_${SLURM_JOB_ID}.log &
PID2=$!

# 3) Terminal-student penalty=50.0, replay+critic reset
CUDA_VISIBLE_DEVICES=2 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type dense --reward_switch_after_steps ${SWITCH_STEPS_GLOBAL} \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --no_hard_block_lethal \
  --intervention_reward_mode penalty_student --intervention_reward_value 50.0 \
  --intervention_enable_after_steps ${ENABLE_AFTER_STEPS_PER_ENV} \
  --exp_name "E_term_pen50_resetcrit_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_E_term_pen50_resetcrit_${SLURM_JOB_ID}.log &
PID3=$!

# 4) Counterfactual buffer penalty=50.0, replay+critic reset
CUDA_VISIBLE_DEVICES=3 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --reward_type dense --reward_switch_after_steps ${SWITCH_STEPS_GLOBAL} \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --no_hard_block_lethal \
  --cf_buffer_enable --cf_sample_ratio 0.5 --cf_q_weight 1.0 --cf_penalty 50.0 \
  --intervention_reward_mode none \
  --intervention_enable_after_steps ${ENABLE_AFTER_STEPS_PER_ENV} \
  --exp_name "E_cf_pen50_resetcrit_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_E_cf_pen50_resetcrit_${SLURM_JOB_ID}.log &
PID4=$!

echo "PIDs: $PID1 $PID2 $PID3 $PID4"
wait $PID1; E1=$?
wait $PID2; E2=$?
wait $PID3; E3=$?
wait $PID4; E4=$?

if [ $E1 -ne 0 ] || [ $E2 -ne 0 ] || [ $E3 -ne 0 ] || [ $E4 -ne 0 ]; then
  echo "One or more experiments failed: $E1 $E2 $E3 $E4"; exit 1
fi

echo "Batch E (critic reset variants) completed."
