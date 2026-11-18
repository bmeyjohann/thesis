"""Read-only validation of completed modality checkpoints and evaluation cohorts."""
import json
from pathlib import Path
import torch
root=Path('artifacts/target_modality_gate_20260914')
manifest=json.loads(Path('artifacts/rgb_height_extended_20260913/manifest.json').read_text())
expected=[r['seed'] for r in manifest['cases']]
rgb_weights=None
for mode in ('height_scan','rgb','rgb_height','depth_height'):
    train=json.loads((root/mode/'summary.json').read_text())
    checkpoint=torch.load(train['checkpoint'],map_location='cpu',weights_only=False)
    data=json.loads((root/'evaluations'/mode/'results.json').read_text())
    assert train['steps']==4500 and train['updates']==3997
    assert checkpoint['args']['navigation_observation_mode']==mode
    assert train['gate']=='hazard_progress'
    assert data['args']['episode_length_s']==180 and not data['benchmark']['hazard_progress']
    assert data['layout']==manifest['layout']
    assert [r['seed'] for r in data['episodes']]==expected
    assert all(r['intervention_fraction']==0 for r in data['episodes'])
    assert all(torch.isfinite(v).all() for v in checkpoint['actor_state_dict'].values())
    if mode!='height_scan':
        assert checkpoint['args']['navigation_encoder_frozen']
        weights=checkpoint['navigation_vision']['encoders']
        if mode=='rgb':rgb_weights=weights['mono_rgb']
        if mode=='rgb_height':
            assert all(torch.equal(v,weights['mono_rgb'][k]) for k,v in rgb_weights.items())
    print('PASS',mode,'4500 decisions, 3997 updates, correct modality, matched 24-goal teacher-free cohort, finite actor')
