#!/bin/bash
#SBATCH --job-name=subgoal_curriculum
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/subgoal_curriculum_%j.out
#SBATCH --error=logs/subgoal_curriculum_%j.err

# Three experiments:
# 1) Dense without subgoals
# 2) Dense with reward-only subgoal shaping
# 3) Curriculum: Stage1 dense+subgoals, Stage2 sparse-only (hard cutoff)

set -e
mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

ENV_ID=${ENV_ID:-pointmaze-arena-danger-wall-v0}
TOTAL_STEPS=${TOTAL_STEPS:-5000000}
NUM_ENVS=${NUM_ENVS:-64}
STAGE1_STEPS=${STAGE1_STEPS:-3000000}  # hard cutoff
DENSE_SCALE=${DENSE_SCALE:-0.01}
SHAPING_COEF=${SHAPING_COEF:-1.0}
SHAPING_GAMMA=${SHAPING_GAMMA:-0.99}

BASE_ARGS="--num_envs ${NUM_ENVS} \
--total_timesteps ${TOTAL_STEPS} \
--include_goal \
--config config/ogbench_config.yaml \
--env_name ${ENV_ID}"

echo "Launching 3 experiments on ${ENV_ID} with ${NUM_ENVS} envs each"

# 1) Dense, no subgoals
CUDA_VISIBLE_DEVICES=0 python train_rsl_rl_integrated.py ${BASE_ARGS} \
  --reward_type dense \
  --dense_reward_scale ${DENSE_SCALE} \
  --experiment_name "${ENV_ID}_dense_nosub_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_dense_nosub_${SLURM_JOB_ID}.log &
PID1=$!

# 2) Dense + subgoal shaping (reward-only)
CUDA_VISIBLE_DEVICES=1 python train_rsl_rl_integrated.py ${BASE_ARGS} \
  --reward_type dense \
  --dense_reward_scale ${DENSE_SCALE} \
  --use_subgoal_shaping \
  --subgoal_shaping_coef ${SHAPING_COEF} \
  --subgoal_shaping_gamma ${SHAPING_GAMMA} \
  --experiment_name "${ENV_ID}_dense_subgoals_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_dense_subgoals_${SLURM_JOB_ID}.log &
PID2=$!

# 3) Curriculum: Stage1 dense+subgoals, Stage2 sparse-only (hard cut)
CUDA_VISIBLE_DEVICES=2 python train_rsl_rl_integrated.py ${BASE_ARGS} \
  --reward_type dense \
  --dense_reward_scale ${DENSE_SCALE} \
  --use_subgoal_shaping \
  --subgoal_shaping_coef ${SHAPING_COEF} \
  --subgoal_shaping_gamma ${SHAPING_GAMMA} \
  --curriculum_stage1_steps ${STAGE1_STEPS} \
  --experiment_name "${ENV_ID}_curriculum_${SLURM_JOB_ID}" \
  2>&1 | tee logs/${ENV_ID}_curriculum_${SLURM_JOB_ID}.log &
PID3=$!

echo "PIDs: $PID1 $PID2 $PID3"
wait $PID1; E1=$?
wait $PID2; E2=$?
wait $PID3; E3=$?

if [ $E1 -ne 0 ] || [ $E2 -ne 0 ] || [ $E3 -ne 0 ]; then
  echo "One or more experiments failed: $E1 $E2 $E3"; exit 1
fi

echo "All subgoal experiments completed."
