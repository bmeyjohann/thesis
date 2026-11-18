"""Frozen reconstruction-CNN transfer for explicit browser navigation modalities."""
from dataclasses import asdict
from functools import lru_cache
import hashlib
from pathlib import Path
import math

import torch
from torch import nn
import numpy as np

VARIANTS = {
    "php_depth": (False, ("php_depth",)),
    "height_scan": (True, ()),
    "rgb": (False, ("mono_rgb",)),
    "rgb_height": (True, ("mono_rgb",)),
    "depth": (False, ("depth",)),
    "depth_height": (True, ("depth",)),
    "rgb_depth": (False, ("mono_rgb", "depth")),
    "stereo_rgb": (False, ("stereo_rgb",)),
    "stereo_rgb_height": (True, ("stereo_rgb",)),
}
DEFAULT_ENCODER = str(Path(__file__).resolve().parent / "artifacts/unitree_multimodal_target/"
    "gentle_gru_reconstruction_shared_seed1_100k_20260903_v3/student_100000.pt")


def add_navigation_camera_args(parser):
    parser.add_argument("--navigation-observation-mode", choices=tuple(VARIANTS), default="height_scan")
    parser.add_argument("--navigation-encoder-checkpoint", default=DEFAULT_ENCODER)
    parser.add_argument("--navigation-image-profile", choices=('legacy', 'stabilized_goal_v1', 'stabilized_goal_v2', 'stabilized_goal_v3'), default='legacy')
    parser.add_argument("--navigation-feature-lags", type=int, nargs="+", default=None,
                        help="Exact decision ages for fused perception features; context stays current. Requires scan-history 1 and action-history 0.")


def camera_enabled(args):
    return getattr(args, "navigation_observation_mode", "height_scan") != "height_scan"


@lru_cache(maxsize=8)
def _checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def transfer_payload(args):
    if str(getattr(args, "model_path", "")):
        saved = _checkpoint(str(args.model_path))
        if "navigation_vision" not in saved:
            raise ValueError("A height-only navigation checkpoint cannot initialize a camera policy; start a new head")
        payload = saved["navigation_vision"]
        if payload["mode"] != args.navigation_observation_mode:
            raise ValueError("Navigation checkpoint sensor mode differs from the requested variant")
        return payload
    path = Path(args.navigation_encoder_checkpoint).expanduser().resolve()
    if args.navigation_observation_mode == "php_depth":
        from unitree_php_depth import PHPDepthEncoder, PHP_SHA256
        encoder = PHPDepthEncoder.from_release(path)
        return dict(version=1, mode="php_depth", encoder_kind="php_depth_v1",
                    encoders={"php_depth": encoder.state_dict()},
                    camera={"width": 106, "height": 60, "max_depth": 3.0},
                    source=str(path), source_sha256=PHP_SHA256,
                    frozen=True, recurrent_transfer=False)
    donor = _checkpoint(str(path))
    if donor.get("architecture") != "gru_reconstruction":
        raise ValueError("Expected the shared GRU reconstruction donor checkpoint")
    _, names = VARIANTS[args.navigation_observation_mode]
    weights = {name: {k.removeprefix(f"encoders.{name}."): v
                       for k, v in donor["student_state_dict"].items()
                       if k.startswith(f"encoders.{name}.")} for name in names}
    return dict(version=1, mode=args.navigation_observation_mode, encoders=weights,
                camera=donor["camera"], source=str(path),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                frozen=True, recurrent_transfer=False)


def configure_navigation_cameras(env_cfg, args):
    if not camera_enabled(args):
        return
    from unitree_nav_vision import UnitreeVisionCfg, attach_vision_sensors
    if getattr(args, "policy_encoder", "mlp") != "mlp":
        raise ValueError("Transferred camera features currently require the MLP navigation head")
    payload = transfer_payload(args)
    # Saved visual preprocessing is authoritative, including legacy checkpoints.
    args.navigation_image_profile = (payload.get('image_profile', 'legacy') if getattr(args, 'model_path', '')
                                     else getattr(args, 'navigation_image_profile', 'legacy'))
    if args.navigation_image_profile != 'legacy' and args.goal_encoding != 'distance_bearing':
        raise ValueError('Stabilized visual goal profile requires distance_bearing goals')
    names = VARIANTS[args.navigation_observation_mode][1]
    stereo = "stereo_rgb" in names
    mode = "stereo_rgb" if stereo else ("rgbd" if "depth" in names else "mono_rgb")
    camera = payload["camera"]
    cfg = UnitreeVisionCfg(mode=mode, width=camera["width"], height=camera["height"],
                          max_depth_m=camera["max_depth"])
    php = args.navigation_observation_mode == "php_depth"
    if php:
        cfg = UnitreeVisionCfg(mode="rgbd", width=106, height=60, max_depth_m=3.0,
            fovy=math.degrees(2 * math.atan(math.tan(math.radians(89.5 / 2)) * 60 / 106)),
            pitch_down_deg=27.0, camera_x_m=.01, camera_z_m=.44)
    if "rig" in payload:
        cfg = UnitreeVisionCfg(**{k:v for k,v in payload["rig"].items() if k != "left_camera_y_m"})
    args.navigation_camera_names = list(attach_vision_sensors(env_cfg, cfg))
    # The donor's mono/depth branches used the LEFT camera of the stereo rig.
    for sensor in env_cfg.scene.sensors:
        if sensor.name in args.navigation_camera_names:
            if not stereo:
                sensor.pos = (cfg.camera_x_m, .01 if php else -cfg.stereo_baseline_m / 2, cfg.camera_z_m)
            if names in (("depth",), ("php_depth",)):
                sensor.data_types = ("depth",)
            elif args.navigation_image_profile in ('stabilized_goal_v2', 'stabilized_goal_v3'):
                # Same camera rays, with range available for occlusion-aware goal identity.
                sensor.data_types = ('rgb', 'depth')
    args.navigation_camera_config = asdict(cfg)
    if args.navigation_image_profile in ('stabilized_goal_v2', 'stabilized_goal_v3'):
        from unitree_visual_goal import MARKER, configure_marker
        args.navigation_goal_marker_config = dict(payload.get('goal_marker', MARKER))
        configure_marker(env_cfg, args.navigation_goal_marker_config)
    args.navigation_camera_config["left_camera_y_m"] = .01 if php else -cfg.stereo_baseline_m / 2
    args.navigation_encoder_sha256 = payload["source_sha256"]
    args.navigation_encoder_frozen = True
    print(f"[navigation-vision] mode={args.navigation_observation_mode} source={payload['source']} "
          f"sha256={payload['source_sha256']} frozen=True cameras={args.navigation_camera_names} "
          f"resolution={cfg.width}x{cfg.height}; image_profile={args.navigation_image_profile}; low-level controller unchanged", flush=True)


class NavigationCameraFeatures(nn.Module):
    def __init__(self, args, env):
        super().__init__()
        from unitree_multimodal_locomotion import ImageEncoder
        self.args, self.env = args, env
        self.include_scan, self.names = VARIANTS[args.navigation_observation_mode]
        self.payload = {**transfer_payload(args), "rig": args.navigation_camera_config,
                        "image_profile": getattr(args, 'navigation_image_profile', 'legacy')}
        self.visual_goal = self.payload['image_profile'] in ('stabilized_goal_v1', 'stabilized_goal_v2', 'stabilized_goal_v3')
        self.scene_goal = self.payload['image_profile'] in ('stabilized_goal_v2', 'stabilized_goal_v3')
        if self.scene_goal:
            self.payload['goal_marker'] = dict(args.navigation_goal_marker_config)
        if self.visual_goal:
            from unitree_nav_vision import camera_quat
            from unitree_camera_observation import quaternion_matrix
            robot = env.env.unwrapped.scene['robot']
            ids, _ = robot.find_bodies('torso_link')
            self.torso_id = int(ids[0])
            rig = self.payload['rig']
            self.mount_rotation = quaternion_matrix(torch.tensor([camera_quat(rig['pitch_down_deg'], rig['yaw_deg'])],
                                                        device=args.device, dtype=torch.float32))
        self.encoders = nn.ModuleDict()
        for name in self.names:
            state = self.payload["encoders"][name]
            if name == "php_depth":
                from unitree_php_depth import PHPDepthEncoder
                self.encoders[name] = PHPDepthEncoder()
            else:
                self.encoders[name] = ImageEncoder(state["network.0.weight"].shape[1],
                                                  state["network.8.weight"].shape[0])
            self.encoders[name].load_state_dict(state, strict=True)
        self.to(args.device).eval().requires_grad_(False)
        self.last_frames = {}
        self.display_frames = {}
        self.captures = 0

    @torch.no_grad()
    def transform(self, stacked):
        if self.scene_goal:
            from unitree_visual_goal import update_marker
            manager = self.env.env.unwrapped
            if update_marker(manager, height=self.payload['goal_marker']['height']):
                # Goal repair/resampling can occur after the normal sense pass.
                # Refresh only then; the UI never requests its own camera render.
                manager.sim.sense()
                for name in self.args.navigation_camera_names:
                    manager.scene.sensors[name].update(0.)
        sensors = [self.env.env.unwrapped.scene.sensors.get(n).data
                   for n in self.args.navigation_camera_names]
        inputs, frames = {}, {}
        validity = None
        if self.visual_goal:
            from unitree_camera_observation import quaternion_matrix, stabilize, goal_image
            robot = self.env.env.unwrapped.scene['robot']
            torso_rotation = quaternion_matrix(robot.data.body_link_quat_w[:, self.torso_id])
            rotation = torso_rotation @ self.mount_rotation
            rig = self.payload['rig']
            def rectify(value, **kwargs):
                return stabilize(value, rotation, fovy=rig['fovy'], pitch_down=rig['pitch_down_deg'], **kwargs)
        goal_mask = None
        if self.scene_goal:
            from unitree_visual_goal import visible_cylinder_mask
            parameters = self.payload['goal_marker']
            ranges = sensors[0].depth[..., 0]
            offset = ranges.new_tensor([rig['camera_x_m'], rig['left_camera_y_m'], rig['camera_z_m']])
            position = robot.data.body_link_pos_w[:, self.torso_id]+torch.einsum('bij,j->bi', torso_rotation, offset)
            center = torch.zeros_like(position)
            center[:, :2] = manager.command_manager._terms['pose']._goal_pos_w[:, :2]
            center[:, 2] = manager.scene.env_origins[:, 2]+parameters['height']/2
            goal_mask = visible_cylinder_mask(ranges, position, rotation, center,
                fovy=rig['fovy'], radius=parameters['radius'], height=parameters['height'])
            goal_mask, _ = rectify(goal_mask, depth=True, far=0.)
        if "mono_rgb" in self.names or "stereo_rgb" in self.names:
            left = sensors[0].rgb
            if left is None:
                raise RuntimeError("RGB sensor is not delivering images")
            frames["vision_rgb"] = left[0].cpu().numpy().copy()
            inputs["mono_rgb"] = left.permute(0, 3, 1, 2).float() / 255
            if self.visual_goal:
                inputs['mono_rgb'], validity = rectify(inputs['mono_rgb'])
            if goal_mask is not None:
                color = inputs['mono_rgb'].new_tensor([.03, 1., .08])[None, :, None, None]
                inputs['mono_rgb'] = torch.where(goal_mask > .5, color, inputs['mono_rgb'])
                self.last_goal_mask = goal_mask > .5
                frames['vision_goal_ray_range'] = sensors[0].depth[0].cpu().numpy().copy()
        if "stereo_rgb" in self.names:
            right = sensors[1].rgb
            if right is None:
                raise RuntimeError("Right stereo camera is not delivering images")
            frames["vision_right_rgb"] = right[0].cpu().numpy().copy()
            right_input = right.permute(0,3,1,2).float()/255
            if self.visual_goal:
                right_input, _ = rectify(right_input)
            inputs["stereo_rgb"] = torch.cat((inputs["mono_rgb"], right_input), dim=1)
        if "depth" in self.names or "php_depth" in self.names:
            raw = sensors[0].depth
            if raw is None:
                raise RuntimeError("Depth sensor is not delivering images")
            if "php_depth" in self.names:
                from unitree_php_depth import range_to_axial
                raw = range_to_axial(raw[..., 0], self.payload["rig"]["fovy"])[..., None]
            depth = torch.nan_to_num(raw, nan=self.payload["camera"]["max_depth"],
                                     posinf=self.payload["camera"]["max_depth"], neginf=0.0)
            frames["vision_depth"] = depth[0].cpu().numpy().copy()
            if self.visual_goal:
                corrected, validity = rectify(depth.permute(0,3,1,2), depth=True,
                    axial='php_depth' in self.names, far=self.payload['camera']['max_depth'])
                depth = corrected.permute(0,2,3,1)
            if "php_depth" in self.names:
                from unitree_php_depth import prepare_php_depth
                inputs["php_depth"] = prepare_php_depth(depth[..., 0])
                if goal_mask is not None:
                    from unitree_visual_goal import annotate_php_goal
                    inputs['php_depth'], self.last_goal_mask = annotate_php_goal(inputs['php_depth'], goal_mask)
            else:
                inputs["depth"] = depth.permute(0,3,1,2) / self.payload["camera"]["max_depth"]
        if self.visual_goal:
            for name in self.names:
                value = inputs[name]
                offset = -.5 if name == 'php_depth' else 0.
                if self.payload['image_profile'] == 'stabilized_goal_v3':
                    pass  # Scene goal only; preserve the original image extent.
                elif name == 'stereo_rgb':
                    value = torch.cat([goal_image(v, stacked[:, :9], encoding=self.args.goal_encoding,
                        distance_scale=self.args.goal_distance_scale) for v in value.split(3, 1)], 1)
                else:
                    value = goal_image(value, stacked[:, :9], encoding=self.args.goal_encoding,
                        distance_scale=self.args.goal_distance_scale, low=offset, high=offset+1.)
                inputs[name] = value
                frames['policy_image_'+name] = value[0].permute(1,2,0).cpu().numpy().copy()
            frames['vision_valid_mask'] = validity[0,0].cpu().numpy().copy()
        self.last_encoder_inputs = inputs
        features = torch.cat([self.encoders[name](inputs[name]) for name in self.names], dim=1)
        if self.visual_goal:
            # Explicit low-resolution validity context prevents hidden warp borders.
            mask = validity
            if 'php_depth' in self.names:
                mask = mask[:, :, 2:, 4:-4]
            features = torch.cat((features, torch.nn.functional.adaptive_avg_pool2d(mask, (4,4)).flatten(1)), 1)
        if not torch.isfinite(features).all():
            raise RuntimeError("Camera encoder produced non-finite features")
        history_dim = 3 * self.args.action_history
        scan_end = stacked.shape[1] - history_dim
        blocks = [stacked[:, :9]]
        if self.include_scan:
            blocks.append(stacked[:, 9:scan_end])
        blocks.append(features)
        if history_dim:
            blocks.append(stacked[:, -history_dim:])
        self.last_frames = frames
        if getattr(self.args, 'observation_only', False):
            # Reuse encoder inputs; never render a second camera for the human.
            if 'php_depth' in inputs:
                self.display_frames = {'PHP normalized depth': (inputs['php_depth'][0, 0] + .5).cpu().numpy().copy()}
            else:
                self.display_frames = {'RGB encoder input': inputs['mono_rgb'][0].permute(1,2,0).cpu().numpy().copy()}
        self.captures += 1
        return torch.cat(blocks, dim=1)

    def checkpoint_payload(self):
        return {**self.payload, "encoders": {name: {k:v.detach().cpu() for k,v in encoder.state_dict().items()}
                                             for name,encoder in self.encoders.items()}}

    def preview_image(self):
        panels = []
        for name in ("vision_rgb", "vision_right_rgb", "vision_depth"):
            if name not in self.last_frames:
                continue
            frame = self.last_frames[name]
            if name == "vision_depth":
                near = 1 - np.clip(frame[..., 0] / self.payload['camera']['max_depth'], 0, 1)
                frame = (255 * np.stack((near, np.sqrt(near), 1-near), axis=-1)).astype(np.uint8)
            panels.append(frame)
        return np.concatenate(panels, axis=1)
