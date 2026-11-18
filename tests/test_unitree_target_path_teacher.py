import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from unitree_target_terrain import generate_target_terrain,_material_at
from unitree_target_path_teacher import TargetPathTeacher

def test_saved_sand_detour():
    manifest=json.loads((ROOT/'artifacts/rgb_height_extended_20260913/manifest.json').read_text())
    teacher=TargetPathTeacher(generate_target_terrain(7,preset='challenge',arena_size=24))
    case=next(c for c in manifest['cases'] if c['seed']==281)
    path=teacher.plan(np.array(case['start']),np.array(case['goal']))
    assert path is not None
    assert all(not teacher.hard[teacher.cell(point)] for point in path)
    assert all(_material_at(teacher.layout,*point)!='sand' for point in path)
    for yaw in np.linspace(-np.pi,np.pi,11):
        action=teacher.action(np.array(case['start']),yaw)
        assert np.isfinite(action).all() and np.linalg.norm(action)<=1.00001
    assert np.linalg.norm(teacher.action(np.array(case['goal']),0))==0
    assert teacher.plan(np.array([11.99,11.99]),np.array(case['goal'])) is None

if __name__=='__main__':
    test_saved_sand_detour()
    print('PASS target path planner: sand detour, clearance, bounded commands, blocked-start abstention')
