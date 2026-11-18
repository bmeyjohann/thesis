"""Bounded offline teacher-imitation screen, distinct from online intervention RL."""
import argparse
import json
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

VARIANTS = ('height', 'rgb_frozen', 'rgb_height_frozen', 'rgb_height_unfrozen',
            'rgb_height_random', 'depth_height_frozen', 'resnet18_height_frozen')


class ScreenPolicy(nn.Module):
    def __init__(self, variant, base_dim, donor=None, mask_action_history=False):
        super().__init__()
        from unitree_multimodal_locomotion import ImageEncoder
        self.variant = variant
        self.mask_action_history = mask_action_history
        self.has_height = variant != 'rgb_frozen'
        self.image_kind = 'depth' if variant.startswith('depth') else 'rgb'
        self.encoder = None
        feature_dim = 0
        if variant != 'height':
            if variant.startswith('resnet18'):
                from torchvision.models import resnet18, ResNet18_Weights
                self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if donor is not None else None)
                self.encoder.fc = nn.Identity()
                self.preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
                feature_dim = 512
            else:
                self.encoder = ImageEncoder(1 if self.image_kind == 'depth' else 3, 64)
                feature_dim = 64
                if donor is not None and not variant.endswith('random'):
                    prefix = 'encoders.' + ('depth' if self.image_kind == 'depth' else 'mono_rgb') + '.'
                    self.encoder.load_state_dict({k[len(prefix):]:v for k,v in donor['student_state_dict'].items() if k.startswith(prefix)})
            self.encoder.requires_grad_(variant.endswith(('unfrozen', 'random')))
        context_dim = base_dim if self.has_height else 24
        self.head = nn.Sequential(nn.Linear(context_dim+feature_dim,128),nn.LayerNorm(128),nn.ELU(),
                                  nn.Linear(128,128),nn.ELU(),nn.Linear(128,3),nn.Tanh())
        self.register_buffer('base_mean',torch.zeros(base_dim))
        self.register_buffer('base_std',torch.ones(base_dim))

    def train(self, mode=True):
        super().train(mode)
        if self.encoder is not None and not any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.eval()
        return self

    def image_features(self,rgb,depth):
        image = depth.permute(0,3,1,2).float()/5 if self.image_kind == 'depth' else rgb.permute(0,3,1,2).float()/255
        if self.variant.startswith('resnet18'):
            image = self.preprocess(image)
        return self.encoder(image)

    def forward(self, base, rgb, depth, features=None):
        normalized = (base-self.base_mean)/self.base_std
        if self.mask_action_history:
            normalized = torch.cat((normalized[:,:-15],torch.zeros_like(normalized[:,-15:])),1)
        context = normalized if self.has_height else torch.cat((normalized[:,:9],normalized[:,-15:]),1)
        blocks = [context]
        if self.encoder is not None:
            blocks.append(self.image_features(rgb,depth) if features is None else features)
        action = self.head(torch.cat(blocks,1))
        return action/action.norm(dim=1,keepdim=True).clamp_min(1)


def load_policy(path, device):
    saved = torch.load(path,map_location='cpu',weights_only=False)
    model = ScreenPolicy(saved['variant'],saved['base_dim'],mask_action_history=saved.get('mask_action_history',False))
    model.load_state_dict(saved['state_dict'])
    return model.to(device).eval()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1])
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    p.add_argument('--mask-action-history',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    print('[encoder-screen] offline BC; resolved '+str(vars(args)),flush=True)
    torch.set_num_threads(1)
    device='cuda'
    assert torch.cuda.is_available()
    from unitree_nav_camera_features import DEFAULT_ENCODER
    donor=torch.load(DEFAULT_ENCODER,map_location='cpu',weights_only=False)
    files=sorted(args.data.glob('episode_*.npz'))
    good=[]
    for path in files:
        with np.load(path) as values:
            if bool(values['success']):
                good.append((path,{k:values[k].copy() for k in ('base','action','vision_rgb','vision_depth')}))
    assert len(good)>=8,'Need at least eight successful complete teacher episodes'
    validation=set(range(0,len(good),5))
    def group(indices):
        return {key:torch.from_numpy(np.concatenate([good[i][1][key] for i in indices])) for key in good[0][1]}
    train=group([i for i in range(len(good)) if i not in validation])
    val=group(sorted(validation))
    args.output.mkdir(parents=True,exist_ok=args.resume)
    split=dict(train=[str(good[i][0]) for i in range(len(good)) if i not in validation],validation=[str(good[i][0]) for i in sorted(validation)],
               excluded_failed=len(files)-len(good),training_steps=len(train['base']),validation_steps=len(val['base']),objective='offline BC, successful teacher episodes only')
    split['sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    split['validation_mean_action_mse']=float((val['action']-train['action'].mean(0)).square().mean())
    split['validation_zero_action_mse']=float(val['action'].square().mean())
    environment=json.loads((args.data/'results.json').read_text()) if (args.data/'results.json').exists() else None
    if args.resume and (args.output/'split.json').exists():
        assert json.loads((args.output/'split.json').read_text())==split,'Resume dataset or split changed'
    else:
        (args.output/'split.json').write_text(json.dumps(split,indent=2))
    results=json.loads((args.output/'results.json').read_text()) if args.resume and (args.output/'results.json').exists() else []
    for seed in args.seeds:
        for variant in args.variants:
            previous=[r for r in results if r['variant']==variant and r['seed']==seed]
            if previous:
                saved=torch.load(previous[0]['checkpoint'],map_location='cpu',weights_only=False)
                assert saved.get('mask_action_history',False)==args.mask_action_history
                assert len(previous[0]['curve'])==args.epochs
                print(f'[encoder-screen] preserving completed {variant} seed={seed}',flush=True)
                continue
            torch.manual_seed(seed)
            model=ScreenPolicy(variant,train['base'].shape[1],donor,args.mask_action_history).to(device)
            model.base_mean.copy_(train['base'].float().mean(0).to(device))
            model.base_std.copy_(train['base'].float().std(0).clamp_min(.05).to(device))
            initial={k:v.detach().clone() for k,v in model.encoder.state_dict().items()} if model.encoder else {}
            optimizer=torch.optim.Adam([v for v in model.parameters() if v.requires_grad],lr=3e-4)
            generator=torch.Generator().manual_seed(seed)
            curve=[]
            start=time.perf_counter()
            frozen=model.encoder is not None and not any(p.requires_grad for p in model.encoder.parameters())
            datasets=[]
            for source in (train,val):
                if frozen:
                    model.eval()
                    with torch.no_grad():
                        features=torch.cat([model.image_features(source['vision_rgb'][i:i+64].to(device),
                            source['vision_depth'][i:i+64].to(device)).cpu() for i in range(0,len(source['base']),64)])
                    datasets.append(dict(base=source['base'],action=source['action'],features=features))
                else:
                    datasets.append(source)
            training,validation=datasets
            def predict(batch):
                return model(batch['base'],batch.get('vision_rgb'),batch.get('vision_depth'),batch.get('features'))
            for epoch in range(args.epochs):
                model.train()
                losses=[]
                for indices in torch.randperm(len(train['base']),generator=generator).split(128):
                    batch={k:v[indices].to(device) for k,v in training.items()}
                    loss=(predict(batch)-batch['action']).square().mean()
                    optimizer.zero_grad();loss.backward();optimizer.step()
                    losses.append(float(loss.detach()))
                model.eval()
                total=0.
                with torch.no_grad():
                    for indices in torch.arange(len(val['base'])).split(128):
                        batch={k:v[indices].to(device) for k,v in validation.items()}
                        total+=float((predict(batch)-batch['action']).square().sum())
                curve.append(dict(epoch=epoch+1,train_mse=float(np.mean(losses)),validation_mse=total/(3*len(val['base']))))
                print(json.dumps(dict(variant=variant,seed=seed,**curve[-1])),flush=True)
            delta=max([float((v-initial[k]).abs().max()) for k,v in model.encoder.state_dict().items()] or [0.]) if model.encoder else 0.
            if initial:
                assert (delta>0) == variant.endswith(('unfrozen','random')),'Encoder update contract violated'
            checkpoint=args.output/f'{variant}_seed{seed}.pt'
            torch.save(dict(variant=variant,base_dim=train['base'].shape[1],state_dict={k:v.cpu() for k,v in model.state_dict().items()},
                training=vars(args),seed=seed,split=split,camera=donor['camera'],curve=curve,
                mask_action_history=args.mask_action_history,
                environment=environment,donor_source=DEFAULT_ENCODER,
                donor_sha256=hashlib.sha256(Path(DEFAULT_ENCODER).read_bytes()).hexdigest()),checkpoint)
            results.append(dict(variant=variant,seed=seed,curve=curve,encoder_max_change=delta,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                total_parameters=sum(p.numel() for p in model.parameters()),
                gradient_steps=args.epochs*int(np.ceil(len(train['base'])/128)),
                seconds=time.perf_counter()-start,checkpoint=str(checkpoint)))
            (args.output/'results.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
