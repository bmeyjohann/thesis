"""Paired perception diagnostic on the canonical target terrain, not a policy evaluation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch
import torch.nn.functional as F
from unitree_php_depth import PHPDepthEncoder, prepare_php_depth


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(out):
    import mujoco
    from unitree_target_terrain import generate_target_terrain, build_target_terrain_spec
    from unitree_nav_vision import camera_quat
    from PIL import Image
    width, height, baseline, camera_height = 424, 240, .05, 1.2
    hfov, pitch = 89.5, 27.
    fovy = math.degrees(2 * math.atan(math.tan(math.radians(hfov/2)) * height/width))
    fx = width / (2 * math.tan(math.radians(hfov/2)))
    records, depths = [], []
    out.mkdir(parents=True, exist_ok=False)
    frames = out / "frames"
    frames.mkdir()
    for seed in (7, 19):
        layout = generate_target_terrain(seed, preset="challenge", arena_size=24)
        (out / f"layout_{seed}.json").write_text(json.dumps(layout.to_dict(), indent=2))
        spec = build_target_terrain_spec(layout)
        spec.worldbody.add_camera(name="probe_left", pos=(0, 0, camera_height),
                                 quat=camera_quat(pitch), fovy=fovy)
        spec.worldbody.add_camera(name="probe_right", pos=(0, 0, camera_height),
                                 quat=camera_quat(pitch), fovy=fovy)
        model = spec.compile()
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height, width)
        try:
            for oi, obstacle in enumerate(layout.obstacles):
                if obstacle.moving:
                    continue
                approach = obstacle.yaw
                direction = np.array([np.cos(approach), np.sin(approach)])
                for gap in (3., 1.5, .65):
                    center = np.asarray(obstacle.center) - direction * (obstacle.size[0] + gap)
                    for heading_offset in (-25., 0., 25.):
                        yaw = approach + math.radians(heading_offset)
                        quat = np.array(camera_quat(pitch, math.degrees(yaw)))
                        rotation = np.zeros(9)
                        mujoco.mju_quat2Mat(rotation, quat)
                        right_axis = rotation.reshape(3, 3)[:, 0]
                        position = np.r_[center, camera_height]
                        for name, sign in (("probe_left", -1), ("probe_right", 1)):
                            ci = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                            model.cam_pos[ci] = position + sign * baseline / 2 * right_axis
                            model.cam_quat[ci] = quat
                        mujoco.mj_forward(model, data)
                        index = len(records)
                        renderer.disable_depth_rendering()
                        renderer.update_scene(data, camera="probe_left")
                        left = renderer.render().copy()
                        renderer.update_scene(data, camera="probe_right")
                        right = renderer.render().copy()
                        renderer.enable_depth_rendering()
                        renderer.update_scene(data, camera="probe_left")
                        depth = renderer.render().copy()
                        assert np.isfinite(depth).all() and left.std() > 0
                        Image.fromarray(left).save(frames/f"{index:03d}_left.png")
                        Image.fromarray(right).save(frames/f"{index:03d}_right.png")
                        depths.append(depth)
                        records.append(dict(index=index, seed=seed, obstacle=oi, gap_m=gap,
                            heading_offset_deg=heading_offset, position=position.tolist(),
                            quat_wxyz=quat.tolist(), obstacle_center=list(obstacle.center),
                            obstacle_size=list(obstacle.size), obstacle_kind=obstacle.kind))
        finally:
            renderer.close()
    np.savez_compressed(out/"capture.npz", depth_m=np.stack(depths))
    manifest = dict(records=records, camera=dict(width=width,height=height,baseline_m=baseline,
        camera_height_m=camera_height,hfov_deg=hfov,fovy_deg=fovy,pitch_down_deg=pitch,fx=fx,
        depth_convention="axial metres",left_position="minus optical-right baseline/2",
        assumptions="D435i-like FOV, assumed 5cm baseline and 1.2m world camera height; not hardware calibration"),
        scope="Camera-pose sweep on canonical challenge terrain; no robot dynamics or self-occlusion; not policy rollouts",
        selection="Every static obstacle, both seeds, all 3 gaps and 3 headings; no outcome filtering",
        generator_sha256=sha(ROOT/"unitree_target_terrain.py"))
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(f"CAPTURED {len(records)} matched stereo/depth poses",flush=True)


def stereo(out, model_path, runtime):
    sys.path.insert(0, str(runtime))
    import onnxruntime as ort
    from PIL import Image
    ort.preload_dlls()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.log_severity_level = 3
    session = ort.InferenceSession(str(model_path), sess_options=opts,
        providers=[("CUDAExecutionProvider", {"gpu_mem_limit": 3*1024**3}), "CPUExecutionProvider"])
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("Stereo demo requires a working CUDA provider; refusing silent CPU fallback")
    manifest = json.loads((out/"manifest.json").read_text())
    camera = manifest['camera']
    means = np.array([.485,.456,.406],np.float32)[None,:,None,None]
    stds = np.array([.229,.224,.225],np.float32)[None,:,None,None]
    dest=out/"stereo"
    dest.mkdir(exist_ok=True)
    times=[]
    for row in manifest['records']:
        i=row['index']; output=dest/f"{i:03d}.npz"
        if output.exists():
            continue
        feeds={}
        for side in ('left','right'):
            image=np.asarray(Image.open(out/'frames'/f"{i:03d}_{side}.png"))
            h,w=image.shape[:2]
            x=image.transpose(2,0,1)[None].astype(np.float32)/255
            x=np.pad(x,((0,0),(0,0),(0,(-h)%32),(0,(-w)%32)),mode='edge')
            feeds[f'{side}_image']=np.ascontiguousarray((x-means)/stds)
        t=time.perf_counter()
        disparity=session.run(['disparity'],feeds)[0][0,0,:h,:w]
        times.append(time.perf_counter()-t)
        assert disparity.shape==(h,w) and np.isfinite(disparity).all()
        valid=disparity>1e-3
        z=np.divide(camera['fx']*camera['baseline_m'], disparity,
                    out=np.full_like(disparity,np.inf),where=valid)
        np.savez_compressed(output, depth_m=z,disparity=disparity,seconds=times[-1])
        if i%12==0: print(f"STEREO {i+1}/{len(manifest['records'])}, {times[-1]:.3f}s",flush=True)
    (out/'stereo_runtime.json').write_text(json.dumps(dict(providers=session.get_providers(),
        model=str(model_path),sha256=sha(model_path),ort=ort.__version__,
        variant="NVIDIA C-Fast-FoundationStereo public dynamic ONNX; not the Google Drive research checkpoint",
        parameters=14597222, seconds=times),indent=2))


def ridge_probe(x,y,train):
    mean=x[train].mean(0);scale=x[train].std(0).clip(1e-5)
    a=(x-mean)/scale
    a=np.c_[a,np.ones(len(a))]
    reg=np.eye(a.shape[1])*10;reg[-1,-1]=0
    w=np.linalg.solve(a[train].T@a[train]+reg,a[train].T@y[train])
    pred=a@w
    return pred, float(np.mean(np.abs(pred[~train]-y[~train]))),float(1-np.sum((pred[~train]-y[~train])**2)/np.sum((y[~train]-y[~train].mean())**2))


def analyze(out,php_path,runtime):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import html
    sys.path.insert(0,str(runtime))
    import onnxruntime as ort
    from PIL import Image
    torch.set_num_threads(4)
    manifest=json.loads((out/'manifest.json').read_text());rows=manifest['records']
    gt=np.load(out/'capture.npz')['depth_m']
    enc=PHPDepthEncoder.from_release(php_path)
    normalized=prepare_php_depth(torch.from_numpy(gt))
    with torch.no_grad():
        features=enc(normalized).numpy();spatial=enc.spatial(normalized).numpy()
        torch.manual_seed(20261013);random=PHPDepthEncoder().eval();random_features=random(normalized).numpy()
    session=ort.InferenceSession(str(php_path),providers=['CPUExecutionProvider'])
    oracle=np.concatenate([session.run(None,{'depth_image':v.numpy()})[0] for v in normalized[:6]])
    parity=float(np.max(np.abs(oracle-features[:6])))
    assert parity<1e-5,parity
    (out/'onnx_torch_parity.json').write_text(json.dumps(dict(max_abs_error=parity,parameter_count=sum(p.numel() for p in enc.parameters())),indent=2))
    train=np.array([r['seed']==7 for r in rows]);y=np.array([r['gap_m'] for r in rows])
    raw=normalized.numpy().reshape(len(rows),-1)
    mu=raw[train].mean(0);_,_,raw_vt=np.linalg.svd(raw[train]-mu,full_matrices=False)
    raw32=(raw-mu)@raw_vt[:32].T
    fm=features[train].mean(0);_,sv,vt=np.linalg.svd(features[train]-fm,full_matrices=False)
    pc=(features-fm)@vt[:3].T
    metrics={};predictions={}
    for label,x in [('PHP pretrained',features),('Random same CNN',random_features),('Raw depth PCA32',raw32)]:
        pred,mae,r2=ridge_probe(x,y,train);metrics[label]=dict(test_gap_mae_m=mae,test_r2=r2);predictions[label]=pred
    metrics['constant_train_mean']=dict(test_gap_mae_m=float(np.abs(y[~train]-y[train].mean()).mean()))
    stereo_files=[out/'stereo'/f"{r['index']:03d}.npz" for r in rows]
    has_stereo=all(p.exists() for p in stereo_files)
    estimated=None;est_features=None
    if has_stereo:
        estimated=np.stack([np.load(p)['depth_m'] for p in stereo_files])
        with torch.no_grad(): est_features=enc.from_metres(torch.from_numpy(estimated)).numpy()
        valid=(gt>=.3)&(gt<=3)&np.isfinite(estimated)
        px=np.arange(gt.shape[-1])[None,None,:]
        disp_gt=manifest['camera']['fx']*manifest['camera']['baseline_m']/np.maximum(gt,1e-4)
        valid &= px-disp_gt>=0
        errors=np.abs(estimated[valid]-gt[valid])
        est_train=(est_features-features[train].mean(0))/features[train].std(0).clip(1e-5)
        a=(features-features[train].mean(0))/features[train].std(0).clip(1e-5)
        a=np.c_[a,np.ones(len(a))];b=np.c_[est_train,np.ones(len(a))]
        reg=np.eye(a.shape[1])*10;reg[-1,-1]=0
        weights=np.linalg.solve(a[train].T@a[train]+reg,a[train].T@y[train])
        frozen_pred=b@weights
        pred,mae,r2=ridge_probe(est_features,y,train)
        metrics['Stereo PHP refit probe']=dict(test_gap_mae_m=mae,test_r2=r2)
        metrics['Stereo PHP frozen GT probe']=dict(test_gap_mae_m=float(np.abs(frozen_pred[~train]-y[~train]).mean()))
        metrics['stereo_depth']=dict(valid_fraction=float(valid.mean()),mae_m=float(errors.mean()),
            median_abs_error_m=float(np.median(errors)),p95_abs_error_m=float(np.quantile(errors,.95)),
            fraction_error_over_025m=float((errors>.25).mean()),
            mask="GT axial depth .3..3m; finite estimate; horizontal correspondence in bounds; occlusions not excluded",
            latent_rmse=float(np.sqrt(np.mean((features-est_features)**2))))
    np.savez_compressed(out/'features.npz',php=features,spatial=spatial,random=random_features,
        raw_depth_pca32=raw32,pc=pc,train=train,gap=y,stereo_php=est_features)
    summary=dict(metrics=metrics,split='Train entire seed7; test entire seed19; alpha=10 fixed, no test tuning',
        limitations=['Camera sweeps, not closed-loop policy evaluation','No physical calibration or robot self-occlusion',
                    'PCA variance is not semantic understanding','Only two terrain seeds; probe labels are selected-obstacle gap, other obstacles can occlude it',
                    'Depth does not directly observe friction; material colors are simulator conventions'],
        count=len(rows),pretrained_php_sha256=sha(php_path),pca_variance_fraction=(sv[:3]**2/(sv**2).sum()).tolist())
    (out/'analysis.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(15,4.2))
    scat=axes[0].scatter(pc[:,0],pc[:,1],c=y,cmap='viridis',s=20)
    fig.colorbar(scat,ax=axes[0],label='Selected obstacle gap (m)')
    axes[0].set(xlabel='PC1',ylabel='PC2',title='PHP feature PCA; basis fitted on seed 7')
    labels=list(predictions)
    vals=[metrics[k]['test_gap_mae_m'] for k in labels]
    axes[1].barh(labels,vals,color=['#155e75','#94a3b8','#c97c1b'])
    axes[1].axvline(metrics['constant_train_mean']['test_gap_mae_m'],ls='--',c='black',label='constant')
    axes[1].set(xlabel='Held-out gap prediction MAE (m)',title='Seed 19 only; linear probes');axes[1].legend()
    if has_stereo:
        axes[2].scatter(gt[valid][::300],estimated[valid][::300],s=2,alpha=.2)
        axes[2].plot([.3,3],[.3,3],'k--');axes[2].set(xlim=(.3,3),ylim=(0,5),xlabel='True depth (m)',ylabel='Stereo depth (m)',title='Stereo errors before PHP encoder')
    fig.tight_layout();fig.savefig(out/'feature_diagnostics.png',dpi=160);plt.close(fig)
    cards=[]
    for i,row in enumerate(rows):
        ncols=5 if has_stereo else 3
        fig,axes=plt.subplots(1,ncols,figsize=(3.4*ncols,2.8))
        axes[0].imshow(Image.open(out/'frames'/f'{i:03d}_left.png'));axes[0].set_title('Simulator left RGB')
        axes[1].imshow(gt[i],vmin=.3,vmax=3,cmap='viridis_r');axes[1].set_title('True depth [.3,3m]')
        axes[2].imshow(np.sqrt(np.mean(spatial[i]**2,axis=0)),cmap='magma');axes[2].set_title('PHP spatial feature RMS\n8 x 11, before pooling')
        if has_stereo:
            axes[3].imshow(estimated[i],vmin=.3,vmax=3,cmap='viridis_r');axes[3].set_title('Fast-FoundationStereo depth')
            axes[4].plot(features[i],label='true depth');axes[4].plot(est_features[i],label='stereo depth',alpha=.8);axes[4].legend(fontsize=7);axes[4].set_title('Same PHP encoder, 32 features')
        for ax in axes[:min(4,ncols)]:ax.axis('off')
        fig.suptitle(f"Pose {i:03d} | seed {row['seed']} obstacle {row['obstacle']} | gap {row['gap_m']}m, yaw {row['heading_offset_deg']}deg",fontsize=10)
        fig.tight_layout();fig.savefig(out/'frames'/f'{i:03d}_comparison.jpg',dpi=110);plt.close(fig)
        cards.append(f'<article><h2>Pose {i:03d}</h2><img loading="lazy" src="frames/{i:03d}_comparison.jpg"><pre>{html.escape(json.dumps(row))}</pre></article>')
    page='<!doctype html><meta charset="utf-8"><title>Unitree depth transfer: complete paired gallery</title><style>body{font:16px Georgia;background:#f4f1e9;color:#182b32;margin:30px}img{width:100%}article{background:white;padding:18px;margin:20px 0}pre{white-space:pre-wrap;font-size:12px}h1{max-width:900px}</style><h1>Complete paired perception gallery</h1><p>'+html.escape(manifest['scope'])+'</p><p>'+html.escape(manifest['selection'])+'</p><img src="feature_diagnostics.png"><pre>'+html.escape(json.dumps(summary,indent=2))+'</pre>'+''.join(cards)
    (out/'index.html').write_text(page)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['capture','stereo','analyze'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--assets',type=Path,default=ROOT/'artifacts/unitree_depth_transfer_20261013')
    a=p.parse_args()
    if a.phase=='capture':capture(a.out)
    elif a.phase=='stereo':stereo(a.out,a.assets/'models/ffs/c_fast_foundationstereo_bp2_dynbatch_dynhw.onnx',a.assets/'runtime')
    else:analyze(a.out,a.assets/'models/php_student/depth_backbone.onnx',a.assets/'runtime')


if __name__=='__main__':
    main()
