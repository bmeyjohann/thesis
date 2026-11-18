from collections import deque
from types import SimpleNamespace
import numpy as np
from unitree_target_hazard_gate import HazardProgressGate,HazardGateConfig


def gate_fixture():
    gate=HazardProgressGate.__new__(HazardProgressGate)
    gate.config=HazardGateConfig()
    gate.teacher=SimpleNamespace(cell=lambda p:(0,0),plan=lambda p,g:np.array([p,g]),
        action=lambda p,y:np.array([.5,0,0]),path=np.array([[0,0],[10,0]]))
    gate.sand=np.array([[False]])
    gate.sand_distance=np.array([[5.]])
    gate.obstacle_distance=np.array([[5.]])
    gate.reset()
    return gate


def test_stall_worsening_and_success():
    gate=gate_fixture();goal=np.array([10.,0.])
    for i in range(15):
        assert gate.decide(np.zeros(2),0,goal,i*.2)[0] is None
    assert gate.decide(np.zeros(2),0,goal,3.)[1]['reason']=='stalled'
    assert gate.decide(goal,0,goal,3.2)[0] is None
    gate.reset()
    for i in range(9):
        result=gate.decide(np.array([-i*.1,0]),0,goal,i*.2)
    assert result[1]['reason']=='worsening'


def test_sand_hysteresis_and_progress():
    gate=gate_fixture();goal=np.array([10.,0.])
    for i in range(30):
        assert gate.decide(np.array([i*.1,0]),0,goal,i*.2)[0] is None
    gate.sand_distance[:]=.6
    assert gate.decide(np.array([3.,0]),0,goal,6.)[1]['reason']=='near_sand'
    gate.sand_distance[:]=.8
    assert gate.decide(np.array([3.1,0]),0,goal,9.)[0] is not None
    gate.sand_distance[:]=1.1
    assert gate.decide(np.array([3.2,0]),0,goal,9.2)[0] is None


def test_escape_path_is_feasible():
    from unitree_target_path_teacher import TargetPathTeacher
    from unitree_target_terrain import generate_target_terrain
    teacher=TargetPathTeacher(generate_target_terrain(7,preset='challenge',arena_size=24))
    gate=HazardProgressGate(teacher)
    cell=tuple(np.argwhere(gate.sand & ~teacher.hard)[0])
    assert gate._escape_plan(teacher.xy[cell])
    assert all(not teacher.hard[teacher.cell(p)] for p in teacher.path)
    assert gate.sand_distance[teacher.cell(teacher.goal)]>=gate.config.sand_release
