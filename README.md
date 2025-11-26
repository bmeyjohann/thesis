# Repository for paper "Implicitly Learning Traversability from Minimal/Requested Human Intervention"

## Abstract

Navigating challenging terrains without continuous human oversight is critical for deploying legged robots in real-world scenarios. This thesis aims to develop a method for rapidly and cost-effectively teaching robots to traverse unknown environments by leveraging dense human interventions as off-policy corrections. Initially, a naive navigation policy is trained using only the Euclidean distance to the goal as a reward in an environment with varying terrain types. Human corrections are logged whenever the policy chooses poor or unsafe routes. These corrections are then integrated as off-policy data to shape the policy's implicit understanding of terrain traversability, without relying on explicit maps. Later, the agent should decide by itself when human input is needed, with the goal of minimizing intervention frequency and need for continuous human oversight over time. We will evaluate how effectively the policy learns to avoid risky terrain, how quickly it can reduce the need for human oversight, and how well its uncertainty estimates align with actual hazards. While initial experiments will be conducted in a 2D gridworld, this research ultimately aims to inform future applications in quadrupedal legged robots navigating real, unstructured environments.

## Quick Start for gridworld experiments

- Remote training: `source sc_venv_template/activate.sh` then `python train_rsl_rl_integrated.py --env_name [pointmaze-medium-v0, pointmaze-arena-danger-{wall,lethal,sticky,floor}-v0]`
- Local training: `conda activate fasttd3` then `python train_fast_sac_ogbench.py --env_name [pointmaze-medium-v0, pointmaze-arena-danger-{wall,lethal,sticky,floor}-v0]`
- Evaluate: `python eval_interactive.py --model_path models/<file>.pt --env_name [pointmaze-medium-v0, pointmaze-arena-danger-{wall,lethal,sticky,floor}-v0]`
- Submit on SLURM: `bash submit_job.sh [experiment_name].sbatch`

## Apptainer for Isaac Sim experiments

- Create separate directory for apptainer: `mkdir -p /p/home/jusers/meyjohann1/juwels/meyjohann1/apptainer/{cache,tmp,images}`
- Use created cache and tmp dirs (must be absolute paths): `export APPTAINER_TMPDIR="/p/home/jusers/meyjohann1/juwels/meyjohann1/apptainer/tmp" && export APPTAINER_CACHE="/p/home/jusers/meyjohann1/juwels/meyjohann1/apptainer/cache"`
- Get image using: `apptainer pull /p/home/jusers/meyjohann1/juwels/meyjohann1/apptainer/images/isaacsim-5.0.0_base.sif docker://{nvcr.io/nvidia/isaac-sim:5.0.0, nvcr.io/nvidia/isaac-lab:2.2.0}`
- Copy image before applying any modifications: `cp /p/home/jusers/meyjohann1/juwels/meyjohann1/apptainer/images/isaacsim-5.0.0_base.sif /p/home/jusers/meyjohann1/juwels/meyjohann1/apptainer/images/isaacsim-5.0.0.sif`
- Drop into container: `apptainer shell --nv --cleanenv --bind $PWD:/workspace /p/home/jusers/meyjohann1/juwels/meyjohann1/apptainer/images/isaacsim-5.0.0.sif`
- Python must be executed using `/isaac-sim/python.sh`

---

See Repository Guidelines for structure, style, and workflows:
- `AGENTS.md`

---

## SLURM commands

- Start a session: `salloc --partition=booster --account=hai_1074 --gres=gpu:1 --time=01:00:00`
- Enter session: `srun --pty bash -l`

---

## Cluster steps

export UNITREE_ROS_DIR=/absolute/path/to/unitree_ros/unitree_ros
export GIT_PYTHON_REFRESH=quiet
unset CUDA_VISIBLE_DEVICES

source sc_venv_template_isaaclab/activate.sh

source .venv/bin/activate

export XDG_CACHE_HOME=/p/project1/hai_1074/meyjohann1/apptainer/cache \
export XDG_DATA_HOME=/p/project1/hai_1074/meyjohann1/apptainer/data \
export KIT_USER_ROOT=/p/project1/hai_1074/meyjohann1/apptainer/kit \
export KIT_USER_DATA=/p/project1/hai_1074/meyjohann1/apptainer/data \
export KIT_CACHE_DIR=/p/project1/hai_1074/meyjohann1/apptainer/cache \
export KIT_DERIVED_DATA_CACHE=/p/project1/hai_1074/meyjohann1/apptainer/DerivedDataCache \
export KIT_PIP_INSTALL_PATH=/p/project1/hai_1074/meyjohann1/apptainer/pip3-envs \
export OMNI_USER_CACHE_DIR=/p/project1/hai_1074/meyjohann1/apptainer/cache \
export OMNI_KIT_USER_DIR=/p/project1/hai_1074/meyjohann1/apptainer/kit \
export OMNI_KIT_DATA_DIR=/p/project1/hai_1074/meyjohann1/apptainer/data \
export OMNI_KIT_CACHE_DIR=/p/project1/hai_1074/meyjohann1/apptainer/cache

export OMNI_DISABLE_EXTENSIONS="omni.iray.libs,omni.mdl.neuraylib,omni.kit.usd.mdl,carb.scenerenderer-rtx.plugin,omni.rtx.*,omni.hydra.rtx"

_isaac_sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py --headless --task Isaac-Velocity-Flat-Unitree-Go2-v0

---


## Interactive Apptainer session setup

# pick a base you control (fast scratch is ideal)
export ISAAC_ROOT="$HOME/meyjohann1/apptainer"              # or your project scratch
export KIT_USER_ROOT="$ISAAC_ROOT/.kit"          # per-job/per-run is best
mkdir -p "$KIT_USER_ROOT"/{cache,DerivedDataCache,data,pip3-envs,cache/nv_shadercache}

# cache buckets you may want to persist between runs
mkdir -p "$ISAAC_ROOT/isaac-cache"/{ov,matplotlib,fontconfig,nvidia,warp}

# writable “overlay” to shadow read-only omni site-packages if needed
export OVERLAY="$ISAAC_ROOT/overlay"
mkdir -p "$OVERLAY/py310/omni"/{cache,data}
mkdir -p "$OVERLAY/py311/omni"/{cache,data}

export IMG="$ISAAC_ROOT/images/isaacsim-5.0.0.sif"

apptainer shell --nv --cleanenv --writable-tmpfs \
  --bind "$KIT_USER_ROOT:/workspace/.kit:rw" \
  --bind "$ISAAC_ROOT/isaac-cache/ov:/root/.cache/ov:rw" \
  --bind "$ISAAC_ROOT/isaac-cache/matplotlib:/root/.cache/matplotlib:rw" \
  --bind "$ISAAC_ROOT/isaac-cache/fontconfig:/root/.cache/fontconfig:rw" \
  --bind "$ISAAC_ROOT/isaac-cache/nvidia:/root/.cache/nvidia:rw" \
  --bind "$ISAAC_ROOT/isaac-cache/warp:/root/.cache/warp:rw" \
  \
  --bind "$OVERLAY/py310/omni:/workspace/project/env/lib/python3.10/site-packages/omni:rw" \
  --bind "$OVERLAY/py311/omni:/workspace/project/env/lib/python3.11/site-packages/omni:rw" \
  \
  --env OMNI_ENV_PRIVACY_CONSENT=1 \
  --env OMNI_ACCEPT_EULA=Y \
  --env OMNI_KIT_ACCEPT_EULA=Y \
  --env ACCEPT_EULA=Y \
  --env OMNI_SERVER= \
  \
  --env XDG_CACHE_HOME=/workspace/.kit/cache \
  --env XDG_DATA_HOME=/workspace/.kit/data \
  --env KIT_USER_ROOT=/workspace/.kit \
  --env KIT_USER_DATA=/workspace/.kit/data \
  --env KIT_CACHE_DIR=/workspace/.kit/cache \
  --env KIT_DERIVED_DATA_CACHE=/workspace/.kit/DerivedDataCache \
  --env KIT_PIP_INSTALL_PATH=/workspace/.kit/pip3-envs \
  --env OMNI_USER_CACHE_DIR=/workspace/.kit/cache \
  --env OMNI_KIT_USER_DIR=/workspace/.kit \
  --env OMNI_KIT_DATA_DIR=/workspace/.kit/data \
  --env OMNI_KIT_CACHE_DIR=/workspace/.kit/cache \
  --env OMNI_KIT_ARGS="\
    --portable-root=/workspace/.kit \
    --/app/dataDir=/workspace/.kit/data \
    --/app/cacheDir=/workspace/.kit/cache \
    --/app/derivedData/cachePath=/workspace/.kit/DerivedDataCache \
    --/rtx/shaderdb/cachePath=/workspace/.kit/cache/nv_shadercache \
    --/exts/omni.kit.pipapi/enable=0" \
  "$IMG"

LOW_LEVEL_POLICY_PATH=wandb/wandb/offline-run-20251104_235648-1vmwzqxs/files/model_950.pt IsaacLab/isaaclab.sh -p safe-locomotion/scripts/rsl_rl/train.py --headless --task Isaac-Navigation-Flat-Go2-v0 "env.actions.pre_trained_policy_action.policy_path=${LOW_LEVEL_POLICY_PATH}"

  ./isaaclab.sh -p scripts/tutorials/00_sim/log_time.py --headless

  apptainer exec --nv \
  --env ACCEPT_EULA=Y \
  --env PRIVACY_CONSENT=Y \
  --bind $PWD:/workspace \
  --bind ~/meyjohann1/apptainer/official/cache/kit:/isaac-sim/kit/cache:rw \
  --bind ~/meyjohann1/apptainer/official/cache/ov:/root/.cache/ov:rw \
  --bind ~/meyjohann1/apptainer/official/cache/pip:/root/.cache/pip:rw \
  --bind ~/meyjohann1/apptainer/official/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  --bind ~/meyjohann1/apptainer/official/cache/computecache:/root/.nv/ComputeCache:rw \
  --bind ~/meyjohann1/apptainer/official/logs:/root/.nvidia-omniverse/logs:rw \
  --bind ~/meyjohann1/apptainer/official/data:/root/.local/share/ov/data:rw \
  --bind ~/meyjohann1/apptainer/official/documents:/root/Documents:rw \
  --bind ~/meyjohann1/apptainer/official/cache/main:/isaac-sim/.cache:rw \
  --bind ~/meyjohann1/apptainer/official/logs:/isaac-sim/.nvidia-omniverse/logs:rw \
  --bind ~/meyjohann1/apptainer/official/config:/isaac-sim/.nvidia-omniverse/config:rw \
  --bind ~/meyjohann1/apptainer/official/data:/isaac-sim/.local/share/ov/data:rw \
  --bind ~/meyjohann1/apptainer/official/pkg:/isaac-sim/.local/share/ov/pkg:rw \
  ../apptainer/images/isaacsim-5.1.0.sif bash

  apptainer exec --nv --writable-tmpfs \
  --env ACCEPT_EULA=Y \
  --env PRIVACY_CONSENT=Y \
  --bind $PWD:/workspace \
  --bind ~/meyjohann1/apptainer/official/cache/kit:/isaac-sim/kit/cache:rw \
  --bind ~/meyjohann1/apptainer/official/cache/ov:/root/.cache/ov:rw \
  --bind ~/meyjohann1/apptainer/official/cache/pip:/root/.cache/pip:rw \
  --bind ~/meyjohann1/apptainer/official/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  --bind ~/meyjohann1/apptainer/official/cache/computecache:/root/.nv/ComputeCache:rw \
  --bind ~/meyjohann1/apptainer/official/logs:/root/.nvidia-omniverse/logs:rw \
  --bind ~/meyjohann1/apptainer/official/data:/root/.local/share/ov/data:rw \
  --bind ~/meyjohann1/apptainer/official/documents:/root/Documents:rw \
  --bind ~/meyjohann1/apptainer/official/cache/main:/isaac-sim/.cache:rw \
  --bind ~/meyjohann1/apptainer/official/logs:/isaac-sim/.nvidia-omniverse/logs:rw \
  --bind ~/meyjohann1/apptainer/official/config:/isaac-sim/.nvidia-omniverse/config:rw \
  --bind ~/meyjohann1/apptainer/official/data:/isaac-sim/.local/share/ov/data:rw \
  --bind ~/meyjohann1/apptainer/official/pkg:/isaac-sim/.local/share/ov/pkg:rw \
  ../apptainer/images/isaacsim-5.1.0.sif bash

./isaaclab.sh -p scripts/tutorials/00_sim/log_time.py --headless --kit_args="--/persistent/isaac/asset_root/default=_assets_cache/Assets/Isaac/5.1/"
/isaac-sim/python.sh scripts/tutorials/00_sim/log_time.py --headless
/isaac-sim/runheadless.sh --/persistent/isaac/asset_root/default="_assets_cache/Assets/Isaac/5.1/"
/isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py --headless --task Isaac-Ant-v0

------

## Get offline assets working

wget -c https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.1.0.zip.001
wget -c https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.1.0.zip.002
wget -c https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.1.0.zip.003

cat isaac-sim-assets-complete-5.1.0.zip.001 \
    isaac-sim-assets-complete-5.1.0.zip.002 \
    isaac-sim-assets-complete-5.1.0.zip.003 \
  > isaac-sim-assets-complete-5.1.0.zip

module load UnZip

unzip -q isaac-sim-assets-complete-5.1.0.zip -d ../_assets_cache/

### Result:

/p/project1/.../isaacsim_assets/Assets/Isaac/5.1/
 ├─ Isaac/...
 └─ NVIDIA/...

 /isaac-sim/python.sh unitree_rl_lab/scripts/rsl_rl/train.py --headless --kit_args="--/persistent/isaac/asset_root/default=_assets_cache/Assets/Isaac/5.1" --task Unitree-Go2-Velocity
/isaac-sim/python.sh unitree_rl_lab/scripts/rsl_rl/train.py --headless --task Unitree-Go2-Velocity

uv pip install -e IsaacLab/source/isaaclab -e IsaacLab/source/isaaclab_rl -e IsaacLab/source/isaaclab_tasks -e IsaacLab/source/isaaclab_mimic -e IsaacLab/source/isaaclab_assets -e unitree_rl_lab/source/unitree_rl_lab

/p/project1/hai_1074/meyjohann1/isaacsim5.1/kit/data/Kit/Isaac-Sim/5.0/user.config.json

~/meyjohann1/isaacsim5.1/python.sh IsaacLab/scripts/tutorials/00_sim/log_time.py --headless