#!/bin/bash
#SBATCH --job-name=ts_sac_batchH_tol15
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/ts_sac_batchH_tol15_%j.out
#SBATCH --error=logs/ts_sac_batchH_tol15_%j.err

set -e
mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

ENV_ID=${ENV_ID:-pointmaze-arena-danger-lethal-v0}
TOTAL_STEPS=${TOTAL_STEPS:-3000000}
NUM_ENVS=${NUM_ENVS:-64}
TOL_A=${TOL_A:-15}
SWITCH_STEPS_GLOBAL=$((TOTAL_STEPS/3))
ENABLE_AFTER_STEPS_PER_ENV=$((SWITCH_STEPS_GLOBAL/NUM_ENVS))
SAVE_INTERVAL=${SAVE_INTERVAL:-200000}
LOG_INTERVAL=${LOG_INTERVAL:-20000}
DENSE_REWARD_SCALE=${DENSE_REWARD_SCALE:-0.1}
STEP_PENALTY=${STEP_PENALTY:-0.005}
PEN_VALUE=${PEN_VALUE:-0.1}
CF_PENALTY=${CF_PENALTY:-0.1}
CF_Q_WEIGHT=${CF_Q_WEIGHT:-0.5}
BONUS_VALUE=${BONUS_VALUE:-0.1}

BASE_ARGS="--env_name ${ENV_ID} --num_envs ${NUM_ENVS} --total_timesteps ${TOTAL_STEPS} --include_goal \
--save_interval ${SAVE_INTERVAL} --log_interval ${LOG_INTERVAL} --use_wandb \
--dense_reward_scale ${DENSE_REWARD_SCALE} --step_penalty ${STEP_PENALTY} \
--reward_type dense --reward_switch_after_steps ${SWITCH_STEPS_GLOBAL} \
--viz_on_checkpoint --viz_grid_resolution 64 --viz_quiver_stride 1 --viz_device cpu --viz_seed 0"

echo "Launching Batch H (tol=${TOL_A}°, penalty=${PEN_VALUE}) on ${ENV_ID}" 

# 1) Terminal-student penalty, replay reset only
CUDA_VISIBLE_DEVICES=0 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --no_hard_block_lethal \
  --intervention_reward_mode penalty_student --intervention_reward_value ${PEN_VALUE} \
  --intervention_enable_after_steps ${ENABLE_AFTER_STEPS_PER_ENV} \
  --reset_replay_on_switch \
  --exp_name "H_term_tol${TOL_A}_pen${PEN_VALUE}_replay_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_H_term_tol${TOL_A}_pen${PEN_VALUE}_replay_${SLURM_JOB_ID}.log &
PID1=$!

# 2) Terminal-student penalty, replay + critic reset
CUDA_VISIBLE_DEVICES=1 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --no_hard_block_lethal \
  --intervention_reward_mode penalty_student --intervention_reward_value ${PEN_VALUE} \
  --intervention_enable_after_steps ${ENABLE_AFTER_STEPS_PER_ENV} \
  --reset_replay_on_switch --reset_critic_on_switch \
  --exp_name "H_term_tol${TOL_A}_pen${PEN_VALUE}_replaycrit_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_H_term_tol${TOL_A}_pen${PEN_VALUE}_replaycrit_${SLURM_JOB_ID}.log &
PID2=$!

# 3) Counterfactual penalty, replay reset only
CUDA_VISIBLE_DEVICES=2 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --no_hard_block_lethal \
  --cf_buffer_enable --cf_sample_ratio 0.5 --cf_q_weight ${CF_Q_WEIGHT} --cf_penalty ${CF_PENALTY} \
  --intervention_reward_mode none \
  --intervention_enable_after_steps ${ENABLE_AFTER_STEPS_PER_ENV} \
  --reset_replay_on_switch \
  --exp_name "H_cf_tol${TOL_A}_pen${CF_PENALTY}_replay_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_H_cf_tol${TOL_A}_pen${CF_PENALTY}_replay_${SLURM_JOB_ID}.log &
PID3=$!

# 4) Counterfactual penalty, replay + critic reset
CUDA_VISIBLE_DEVICES=3 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --no_hard_block_lethal \
  --cf_buffer_enable --cf_sample_ratio 0.5 --cf_q_weight ${CF_Q_WEIGHT} --cf_penalty ${CF_PENALTY} \
  --intervention_reward_mode none \
  --intervention_enable_after_steps ${ENABLE_AFTER_STEPS_PER_ENV} \
  --reset_replay_on_switch --reset_critic_on_switch \
  --exp_name "H_cf_tol${TOL_A}_pen${CF_PENALTY}_replaycrit_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_H_cf_tol${TOL_A}_pen${CF_PENALTY}_replaycrit_${SLURM_JOB_ID}.log &
PID4=$!

# 5) Terminal penalty + bonus, replay reset only
CUDA_VISIBLE_DEVICES=0 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --no_hard_block_lethal \
  --intervention_reward_mode penalty_student --intervention_reward_value ${PEN_VALUE} \
  --bonus_teacher_value ${BONUS_VALUE} \
  --intervention_enable_after_steps ${ENABLE_AFTER_STEPS_PER_ENV} \
  --reset_replay_on_switch \
  --exp_name "H_term_bonus_tol${TOL_A}_pen${PEN_VALUE}_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_H_term_bonus_tol${TOL_A}_pen${PEN_VALUE}_${SLURM_JOB_ID}.log &
echo "PIDs: $PID1 $PID2 $PID3 $PID4"
wait $PID1; E1=$?
wait $PID2; E2=$?
wait $PID3; E3=$?
wait $PID4; E4=$?

if [ $E1 -ne 0 ] || [ $E2 -ne 0 ] || [ $E3 -ne 0 ] || [ $E4 -ne 0 ]; then
  echo "One or more phase-1 experiments failed: $E1 $E2 $E3 $E4"; exit 1
fi

# Phase 2 runs using freed GPUs
CUDA_VISIBLE_DEVICES=0 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --no_hard_block_lethal \
  --intervention_reward_mode penalty_student --intervention_reward_value ${PEN_VALUE} \
  --bonus_teacher_value ${BONUS_VALUE} \
  --intervention_enable_after_steps ${ENABLE_AFTER_STEPS_PER_ENV} \
  --reset_replay_on_switch \
  --exp_name "H_term_bonus_tol${TOL_A}_pen${PEN_VALUE}_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_H_term_bonus_tol${TOL_A}_pen${PEN_VALUE}_${SLURM_JOB_ID}.log &
PID5=$!

CUDA_VISIBLE_DEVICES=1 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --use_intervention --intervention_mode agent --teacher_type bfs \
  --tolerance_type angle --tolerance_value ${TOL_A} \
  --no_hard_block_lethal \
  --intervention_reward_mode penalty_student --intervention_reward_value ${PEN_VALUE} \
  --bonus_teacher_value ${BONUS_VALUE} \
  --intervention_enable_after_steps ${ENABLE_AFTER_STEPS_PER_ENV} \
  --reset_replay_on_switch --reset_critic_on_switch \
  --exp_name "H_term_bonus_tol${TOL_A}_pen${PEN_VALUE}_crit_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_H_term_bonus_tol${TOL_A}_pen${PEN_VALUE}_crit_${SLURM_JOB_ID}.log &
PID6=$!

wait $PID5; E5=$?
wait $PID6; E6=$?

if [ $E5 -ne 0 ] || [ $E6 -ne 0 ]; then
  echo "One or more phase-2 experiments failed: $E5 $E6"; exit 1
fi

echo "Batch H completed."
