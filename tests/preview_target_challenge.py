"""Paused, non-training browser preview using the saved actor on the new terrain."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import eval_interactive_unitree_nav as interactive

sys.argv = ['preview']
args = interactive.parse_args()
run = ROOT / 'models/unitree_mjlab_nav_human/target_human_20260910_193717'
vars(args).update(json.loads((run/'args.json').read_text()))
args.online_train = False
args.online_total_steps = 0
args.human_dataset_dir = ''
args.model_path = str(run/'final.pt')
args.target_terrain_preset = 'challenge'
args.target_terrain_dynamic_obstacles = True
args.target_terrain_material_resolution = .2
args.target_terrain_ice_friction = .06
args.target_terrain_ice_contact_priority = 1
args.gamepad_debug_console = False
args.gamepad_port = 18794
args.viser_port = 8092
args.start_paused = True
args.autopilot = True
args.viser_camera_mode = 'bird'
args.viser_bird_height = 18
args.viser_max_steps = 100
interactive.run(args)
