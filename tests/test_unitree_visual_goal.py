import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unitree_visual_goal import visible_cylinder_mask, annotate_php_goal


def test_center_ray_matches_cylinder_and_rejects_occlusion():
    rotation = torch.tensor([[[0.,0.,-1.],[-1.,0.,0.],[0.,1.,0.]]])
    camera = torch.tensor([[0.,0.,1.2]])
    center = torch.tensor([[5.,0.,1.2]])
    ranges = torch.full((1,61,81), 100.)
    ranges[0,30,40] = 4.7
    mask = visible_cylinder_mask(ranges,camera,rotation,center,fovy=70,radius=.3,height=2.4)
    assert mask[0,0,30,40] == 1 and mask.sum() == 1
    ranges[0,30,40] = 2.
    assert visible_cylinder_mask(ranges,camera,rotation,center,fovy=70,radius=.3,height=2.4).sum() == 0


def test_far_goal_survives_php_clipping_but_not_outside_mask():
    far = torch.full((1,1,58,87), .5)
    mask = torch.zeros((1,1,60,106))
    mask[:,:,10:50,45:60] = 1
    result, footprint = annotate_php_goal(far,mask)
    assert torch.equal(result[~footprint],far[~footprint])
    assert (result[footprint] == -.5).any() and (result[footprint] == .5).any()
