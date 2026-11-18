"""Live GPU and compiled-geometry checks for the opt-in challenge profile."""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import eval_interactive_unitree_nav as interactive
from unitree_target_terrain import TargetArenaTerrainCfg


def main():
    sys.argv = ['smoke']
    args = interactive.parse_args()
    run = ROOT / 'models/unitree_mjlab_nav_human/target_human_20260910_193717'
    vars(args).update(json.loads((run/'args.json').read_text()))
    args.online_train = False
    args.human_dataset_dir = ''
    args.target_terrain_preset = 'challenge'
    args.target_terrain_dynamic_obstacles = True
    args.target_terrain_material_resolution = .2
    args.target_terrain_ice_friction = .06
    args.target_terrain_ice_contact_priority = 1
    env = interactive.make_env(args, num_envs=1, render=False)
    raw, _, _, _, _ = interactive._reset_until_feasible(args, env)
    u = env.env.unwrapped
    model = u.sim.mj_model
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'target_mover_{i}/obstacle') for i in range(4)]
    assert min(ids) >= 0, ids
    assert np.all(model.body_mocapid[ids] >= 0)
    get_positions = lambda: np.stack([u.scene[f'target_mover_{i}'].data.root_link_pos_w[0].cpu().numpy().copy() for i in range(4)])
    before = get_positions()
    for _ in range(20):
        env.step(torch.zeros((1,3), device=args.device))
    after = get_positions()
    displacement = np.linalg.norm(after-before,axis=1)
    assert np.all(displacement > .01), displacement
    ice = (model.geom_rgba[:, 2] > .8) & (model.geom_rgba[:, 0] < .3)
    assert np.count_nonzero(ice) > 0
    assert np.allclose(model.geom_friction[ice, 0], .06)
    assert np.all(model.geom_priority[ice] == 1)
    gpu_friction = u.sim.model.geom_friction.detach().cpu().numpy()
    assert np.allclose(gpu_friction[..., ice, 0], .06)
    terrain_heights = {}
    for preset in ['balanced','challenge']:
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name='terrain')
        cfg = TargetArenaTerrainCfg(size=(24.,24.), seed=7, preset=preset, material_resolution=.2)
        output = cfg.function(0., spec, np.random.default_rng(7))
        # Non-obstacle elevated gray geometry includes staircase/crest surfaces.
        gray = [g.geom for g in output.geometries if abs(g.geom.rgba[0]-g.geom.rgba[1]) < .13
                and g.geom.pos[2] > .1 and g.geom.size[2] > .05]
        terrain_heights[preset] = max(float(g.pos[2]+g.size[2]) for g in gray)
    assert terrain_heights['challenge'] > terrain_heights['balanced'] * 2
    print(json.dumps(dict(status='PASS', movers=4, displacement_m=displacement.tolist(),
                         ice_geoms=int(ice.sum()), ice_friction=.06, material_resolution=.2,
                         elevated_geometry_height_m=terrain_heights)), flush=True)
    env.close()


if __name__ == '__main__':
    main()
