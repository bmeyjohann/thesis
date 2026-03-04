#!/bin/bash
#SBATCH --job-name=ts_sac_batchO_shared
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/ts_sac_batchO_shared_%j.out
#SBATCH --error=logs/ts_sac_batchO_shared_%j.err

set -e
mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

ENV_ID=${ENV_ID:-pointmaze-arena-danger-wall-v0}
TOTAL_STEPS=${TOTAL_STEPS:-3000000}
NUM_ENVS=${NUM_ENVS:-64}
TOLERANCE=${TOLERANCE:-15}
SWITCH_STEPS=$((TOTAL_STEPS/3))
ENABLE_AFTER_STEPS=$((SWITCH_STEPS/NUM_ENVS))
SAVE_INTERVAL=${SAVE_INTERVAL:-200000}
LOG_INTERVAL=${LOG_INTERVAL:-20000}
DENSE_REWARD_SCALE=${DENSE_REWARD_SCALE:-0.05}
STEP_PENALTY=${STEP_PENALTY:-0.002}
PREF_MARGIN=${PREF_MARGIN:-0.1}
PREF_WEIGHT=${PREF_WEIGHT:-1.0}
THRESHOLDS=${THRESHOLDS:-0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0,2.0,5.0}
SHARED_DIM=${SHARED_DIM:-512}

BASE_ARGS="--env_name ${ENV_ID} --num_envs ${NUM_ENVS} --total_timesteps ${TOTAL_STEPS} --include_goal \
--save_interval ${SAVE_INTERVAL} --log_interval ${LOG_INTERVAL} --use_wandb \
--dense_reward_scale ${DENSE_REWARD_SCALE} --step_penalty ${STEP_PENALTY} \
--reward_type dense --reward_switch_after_steps ${SWITCH_STEPS} \
--post_switch_viz_multiplier 10 --disagreement_thresholds ${THRESHOLDS} \
--viz_on_checkpoint --viz_grid_resolution 64 --viz_quiver_stride 2 --viz_device cpu --viz_seed 0 \
--pref_buffer_enable --pref_capacity 100000 --pref_sample_ratio 0.5 \
--pref_rank_margin ${PREF_MARGIN} --pref_rank_weight ${PREF_WEIGHT} \
--intervention_reward_mode none --bonus_teacher_value 0.0 \
--use_intervention --intervention_mode agent --teacher_type bfs \
--tolerance_type angle --tolerance_value ${TOLERANCE} \
--intervention_enable_after_steps ${ENABLE_AFTER_STEPS} \
--reset_replay_on_switch --reset_critic_on_switch"

echo "Launching Batch O (shared trunk sweep) on ${ENV_ID}"

# 1) Baseline MLP (no shared trunk, 2 critics)
CUDA_VISIBLE_DEVICES=0 python -u train_fast_sac_ogbench_maze.py ${BASE_ARGS} \
  --num_critics 2 \
  --exp_name "O_mlp_base_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_O_mlp_base_${SLURM_JOB_ID}.log &
PID1=$!

# 2) Shared trunk (2 critics)
CUDA_VISIBLE_DEVICES=1 python -u train_fast_sac_ogbench_maze.py ${BASE_ARGS} \
  --arch_shared_trunk --shared_hidden_dim ${SHARED_DIM} --num_critics 2 \
  --exp_name "O_shared2_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_O_shared2_${SLURM_JOB_ID}.log &
PID2=$!

# 3) Shared trunk (3 critics)
CUDA_VISIBLE_DEVICES=2 python -u train_fast_sac_ogbench_maze.py ${BASE_ARGS} \
  --arch_shared_trunk --shared_hidden_dim ${SHARED_DIM} --num_critics 3 \
  --exp_name "O_shared3_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_O_shared3_${SLURM_JOB_ID}.log &
PID3=$!

# 4) Baseline (no shared trunk, 3 critics)
CUDA_VISIBLE_DEVICES=3 python -u train_fast_sac_ogbench_maze.py ${BASE_ARGS} \
  --num_critics 3 \
  --exp_name "O_mlp_3crit_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_sac_O_mlp_3crit_${SLURM_JOB_ID}.log &
PID4=$!

echo "PIDs: $PID1 $PID2 $PID3 $PID4"

wait $PID1; E1=$?
wait $PID2; E2=$?
wait $PID3; E3=$?
wait $PID4; E4=$?

if [ $E1 -ne 0 ] || [ $E2 -ne 0 ] || [ $E3 -ne 0 ] || [ $E4 -ne 0 ]; then
  echo "One or more Batch O experiments failed: $E1 $E2 $E3 $E4"; exit 1
fi

echo "Batch O completed."
