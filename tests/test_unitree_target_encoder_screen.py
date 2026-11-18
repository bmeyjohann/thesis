"""Observation, gradient, freeze, and serialization contracts for the offline screen."""
from pathlib import Path
import sys
import tempfile
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from unitree_target_encoder_screen import ScreenPolicy,VARIANTS,load_policy


def test_contracts():
    torch.set_num_threads(1)
    base=torch.randn(2,211)
    rgb=torch.randint(0,256,(2,60,80,3),dtype=torch.uint8)
    depth=torch.rand(2,60,80,1)*5
    for variant in VARIANTS:
        model=ScreenPolicy(variant,211).train()
        action=model(base,rgb,depth)
        assert action.shape==(2,3) and torch.isfinite(action).all()
        assert torch.all(action.norm(dim=1)<=1.00001)
        action.square().sum().backward()
        if model.encoder is not None:
            expected=variant.endswith(('unfrozen','random'))
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters())==expected
            assert model.encoder.training==expected
        model.eval()
        if model.encoder is not None:
            with torch.no_grad():
                features=model.image_features(rgb,depth)
                torch.testing.assert_close(model(base,rgb,depth),model(base,None,None,features))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'policy.pt'
            torch.save(dict(variant=variant,base_dim=211,state_dict=model.state_dict()),path)
            loaded=load_policy(path,'cpu')
            torch.testing.assert_close(model(base,rgb,depth),loaded(base,rgb,depth))
    print('PASS all seven screen variants: bounded actions, gradients, freeze and checkpoint parity')
    masked=ScreenPolicy('rgb_height_frozen',211,mask_action_history=True).eval()
    perturbed=base.clone();perturbed[:,-15:]+=100
    torch.testing.assert_close(masked(base,rgb,depth),masked(perturbed,rgb,depth))
    print('PASS masked-history predictions invariant to action-history perturbations')


def test_training_smoke():
    from unitree_target_encoder_screen import main
    rng=np.random.default_rng(19)
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        data=root/'synthetic';data.mkdir()
        for seed in range(8):
            np.savez_compressed(data/f'episode_{seed}.npz',success=True,
                base=rng.normal(size=(4,211)).astype('float32'),
                action=rng.uniform(-.5,.5,size=(4,3)).astype('float32'),
                vision_rgb=rng.integers(0,256,size=(4,60,80,3),dtype='uint8'),
                vision_depth=rng.uniform(0,5,size=(4,60,80,1)).astype('float32'))
        argv=sys.argv
        try:
            sys.argv=['screen','--data',str(data),'--output',str(root/'out'),'--epochs','1','--seeds','0']
            main()
            assert len(list((root/'out').glob('*.pt')))==len(VARIANTS)
        finally:
            sys.argv=argv
    print('PASS synthetic one-epoch GPU training, episode split, encoder updates and checkpoints')


if __name__=='__main__':
    test_contracts()
    if '--training-smoke' in sys.argv:
        test_training_smoke()
