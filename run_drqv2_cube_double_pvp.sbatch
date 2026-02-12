#!/bin/bash
#SBATCH --job-name=drqv2_cube_double_pvp
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/drqv2_cube_double_pvp_%j.out
#SBATCH --error=logs/drqv2_cube_double_pvp_%j.err

set -euo pipefail

mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${SLURM_TMPDIR:-/tmp}/xdg_cache_${SLURM_JOB_ID}}
mkdir -p "${XDG_CACHE_HOME}"
export WANDB_DIR=${WANDB_DIR:-$PWD}
mkdir -p "${WANDB_DIR}/wandb"

OGBENCH_ENV=${OGBENCH_ENV:-visual-cube-double-v0}
TOTAL_STEPS=${TOTAL_STEPS:-1200000}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
NUM_EXPL_STEPS=${NUM_EXPL_STEPS:-20000}
SAVE_INTERVAL=${SAVE_INTERVAL:-200000}
EVAL_EVERY=${EVAL_EVERY:-50000}
PROJECT=${PROJECT:-ogbench_cube_double_pvp}
FRAME_STACK=${FRAME_STACK:-3}
ACTION_REPEAT=${ACTION_REPEAT:-1}
FEATURE_DIM=${FEATURE_DIM:-50}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
TOLERANCE_TYPE=${TOLERANCE_TYPE:-l2}
TOLERANCE=${TOLERANCE:-0.3}
TEACHER_TYPE=${TEACHER_TYPE:-cube_plan}
REWARD_TYPE=${REWARD_TYPE:-dense}

BASE_ARGS="--obs_mode pixels --pixel_width 84 --pixel_height 84 \
--total_timesteps ${TOTAL_STEPS} --learning_rate ${LEARNING_RATE} \
--num_expl_steps ${NUM_EXPL_STEPS} --save_interval ${SAVE_INTERVAL} \
--eval_every_frames ${EVAL_EVERY} --project ${PROJECT} \
--frame_stack ${FRAME_STACK} --action_repeat ${ACTION_REPEAT} \
--drq_feature_dim ${FEATURE_DIM} --drq_hidden_dim ${HIDDEN_DIM} \
--nstep 1 \
--use_wandb --viz_on_checkpoint \
--use_intervention --intervention_mode agent --teacher_type ${TEACHER_TYPE} \
--tolerance_type ${TOLERANCE_TYPE} --tolerance_value ${TOLERANCE} \
--reward_type ${REWARD_TYPE}"

echo "Launching cube-double PVP"

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 python -u train_drqv2_pvp_ogbench.py \
  --env_name "${OGBENCH_ENV}" \
  ${BASE_ARGS} \
  --exp_name "pvp_${TEACHER_TYPE}_${OGBENCH_ENV}_${SLURM_JOB_ID}" \
  2>&1 | tee "logs/pvp_${TEACHER_TYPE}_${OGBENCH_ENV}_${SLURM_JOB_ID}.log"
