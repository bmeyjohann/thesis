#!/bin/bash
#SBATCH --job-name=ts_sac_batchM_pref_margin
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/ts_sac_batchM_pref_margin_%j.out
#SBATCH --error=logs/ts_sac_batchM_pref_margin_%j.err

set -e
mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

ENV_ID=${ENV_ID:-pointmaze-arena-danger-wall-v0}
TOTAL_STEPS=${TOTAL_STEPS:-3000000}
NUM_ENVS=${NUM_ENVS:-64}
TOL_A=${TOL_A:-15}
SWITCH_STEPS_GLOBAL=$((TOTAL_STEPS/3))
ENABLE_AFTER_STEPS_PER_ENV=$((SWITCH_STEPS_GLOBAL/NUM_ENVS))
SAVE_INTERVAL=${SAVE_INTERVAL:-200000}
LOG_INTERVAL=${LOG_INTERVAL:-20000}
DENSE_REWARD_SCALE=${DENSE_REWARD_SCALE:-0.05}
STEP_PENALTY=${STEP_PENALTY:-0.002}
BASE_SAMPLE_RATIO=${BASE_SAMPLE_RATIO:-0.5}

BASE_ARGS="--env_name ${ENV_ID} --num_envs ${NUM_ENVS} --total_timesteps ${TOTAL_STEPS} --include_goal \
--save_interval ${SAVE_INTERVAL} --log_interval ${LOG_INTERVAL} --use_wandb \
--dense_reward_scale ${DENSE_REWARD_SCALE} --step_penalty ${STEP_PENALTY} \
--reward_type dense --reward_switch_after_steps ${SWITCH_STEPS_GLOBAL} \
--post_switch_viz_multiplier 10 \
--viz_on_checkpoint --viz_grid_resolution 64 --viz_quiver_stride 2 --viz_device cpu --viz_seed 0 \
--pref_buffer_enable --pref_capacity 100000 --pref_sample_ratio ${BASE_SAMPLE_RATIO} --pref_rank_weight 1.0 \
--use_intervention --intervention_mode agent --teacher_type bfs \
--tolerance_type angle --tolerance_value ${TOL_A} --no_hard_block_lethal \
--intervention_reward_mode none --bonus_teacher_value 0.0 \
--intervention_enable_after_steps ${ENABLE_AFTER_STEPS_PER_ENV} \
--reset_replay_on_switch --reset_critic_on_switch"

echo "Launching Batch M (Preference ranking margin sweep) on ${ENV_ID}"

# Margin variants
MARGINS=(0.05 0.1 0.2 0.3)
WEIGHTS=(1.0 1.0 1.0 1.5)

CUDA_VISIBLE_DEVICES=0 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --pref_rank_margin ${MARGINS[0]} --pref_rank_weight ${WEIGHTS[0]} \
  --exp_name "M_pref_margin_${MARGINS[0]}_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_M_pref_margin_${MARGINS[0]}_${SLURM_JOB_ID}.log &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --pref_rank_margin ${MARGINS[1]} --pref_rank_weight ${WEIGHTS[1]} \
  --exp_name "M_pref_margin_${MARGINS[1]}_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_M_pref_margin_${MARGINS[1]}_${SLURM_JOB_ID}.log &
PID2=$!

CUDA_VISIBLE_DEVICES=2 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --pref_rank_margin ${MARGINS[2]} --pref_rank_weight ${WEIGHTS[2]} \
  --exp_name "M_pref_margin_${MARGINS[2]}_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_M_pref_margin_${MARGINS[2]}_${SLURM_JOB_ID}.log &
PID3=$!

CUDA_VISIBLE_DEVICES=3 python -u train_fast_sac_ogbench.py ${BASE_ARGS} \
  --pref_rank_margin ${MARGINS[3]} --pref_rank_weight ${WEIGHTS[3]} \
  --pref_sample_ratio 0.6 \
  --exp_name "M_pref_margin_${MARGINS[3]}_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_M_pref_margin_${MARGINS[3]}_${SLURM_JOB_ID}.log &
PID4=$!

echo "PIDs: $PID1 $PID2 $PID3 $PID4"
wait $PID1; E1=$?
wait $PID2; E2=$?
wait $PID3; E3=$?
wait $PID4; E4=$?

if [ $E1 -ne 0 ] || [ $E2 -ne 0 ] || [ $E3 -ne 0 ] || [ $E4 -ne 0 ]; then
  echo "One or more Batch M experiments failed: $E1 $E2 $E3 $E4"; exit 1
fi

echo "Batch M completed."

