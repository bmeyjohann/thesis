import sys
from pathlib import Path
import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unitree_eval_contacts import accumulate_contacts


def test_contact_kernel_matches_reference():
    rng = np.random.default_rng(4)
    for device in ['cpu']+(['cuda:0'] if wp.is_cuda_available() else []):
        kind = np.array([0,1,2,0,1,2],dtype=np.int64)
        metrics = wp.zeros(2,dtype=wp.float32,device=device)
        expected = np.zeros(2)
        for _ in range(12):
            ids = rng.integers(-1,6,(100,2),dtype=np.int32)
            world = rng.integers(0,3,100,dtype=np.int32)
            dist = rng.uniform(-.03,.03,100).astype(np.float32)
            n = int(rng.integers(1,100))
            pair = (ids[:,0]>=0)&(ids[:,1]>=0)&(world==0)&(dist<=0)&(np.arange(100)<n)
            kinds = kind[ids.clip(0)]
            pair &= ((kinds[:,0]==1)&(kinds[:,1]==2))|((kinds[:,0]==2)&(kinds[:,1]==1))
            expected[0] += pair.any()
            expected[1] = max(expected[1],np.where(pair,-dist,0).max())
            wp.launch(accumulate_contacts,dim=1,inputs=[wp.array([n],dtype=wp.int32,device=device),
                wp.array(ids,dtype=wp.vec2i,device=device),wp.array(dist,device=device),
                wp.array(world,device=device),wp.array(kind,device=device),metrics],device=device)
        np.testing.assert_allclose(metrics.numpy(),expected,atol=1e-7)
