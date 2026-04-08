from __future__ import annotations

import argparse

from .vr_teleop import DEFAULT_VR_CACHE_PATH, DEFAULT_VR_MAPPING_PATH, DEFAULT_VR_PORT

def build_train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    cube_reward_choices = [
        'sparse_final',
        'sparse_intermediate',
        'dense',
    ]
    # Env
    p.add_argument('--env_name', type=str, default='pointmaze-arena-danger-lethal-v0')
    p.add_argument('--num_envs', type=int, default=64)
    p.add_argument('--max_episode_steps', type=int, default=0,
                   help='Optional positive per-episode step limit override for train/eval env creation (0 uses the env default).')
    p.add_argument(
        '--hold_targets_on_zero_action',
        action='store_true',
        default=False,
        help='Manip only: skip IK recomputation on true zero xyz/yaw actions and hold the previous joint targets.',
    )
    p.add_argument(
        '--noop_action_threshold',
        type=float,
        default=1e-6,
        help='Manip only: absolute threshold for treating xyz/yaw (and gripper, when enabled) as zero for target-hold optimization.',
    )
    p.add_argument(
        '--disable_rotation',
        action='store_true',
        default=False,
        help='Manip only: lock effector/cube/goal yaw and expose a 4D xyz+gripper action space.',
    )
    p.add_argument('--total_timesteps', type=int, default=1_000_000)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--train_render_mode', type=str, default='none', choices=['none', 'human'],
                   help='Optional live rendering during training (recommended only with num_envs=1).')
    p.add_argument(
        '--visualize_intervention_colors',
        action='store_true',
        default=True,
        help='Manip only: tint the live human-render viewer during intervention (default: enabled).',
    )
    p.add_argument(
        '--no_visualize_intervention_colors',
        dest='visualize_intervention_colors',
        action='store_false',
        help='Manip only: disable live intervention tinting in the human-render viewer.',
    )
    p.add_argument('--static_reset_seed', type=int, default=None,
                   help='If set, forces identical reset seed on every episode reset (deterministic static initial states).')
    # Observations
    p.add_argument('--include_goal', action='store_true', default=True)
    p.add_argument('--no_include_goal', dest='include_goal', action='store_false',
                   help='Disable goal concatenation in state observations.')
    p.add_argument('--include_distance', action='store_true', default=False)
    p.add_argument('--include_direction', action='store_true', default=False)
    p.add_argument('--include_velocity', action='store_true', default=False)
    p.add_argument(
        '--include_relative_cube_features',
        action='store_true',
        default=False,
        help=(
            'Manip state only: append target-relative features '
            '(effector->target cube and target cube->goal position/yaw deltas) to observations.'
        ),
    )
    p.add_argument(
        '--relative_only_obs',
        action='store_true',
        default=False,
        help=(
            'Manip state only: keep compact proprioception (effector xyz/yaw + gripper open/contact) '
            'and optional relative cube features while dropping absolute/joint-heavy state fields.'
        ),
    )
    p.add_argument('--goal_marker_color', type=str, default='auto',
                   choices=['auto', 'red', 'green', 'blue'],
                   help='Override maze goal marker color (auto keeps env default)')
    # Algorithm variant / rewards
    p.add_argument(
        '--algo_variant',
        type=str,
        default='own',
        choices=['own', 'pvp', 'eil'],
        help='Training objective family. own = current preference-based method, pvp = proxy value propagation baseline, eil = Expert Intervention Learning-style threshold/ranking objective.',
    )
    p.add_argument('--reward_type', type=str, default='sparse', choices=['sparse', 'dense', 'combined', 'none'])
    p.add_argument('--dense_reward_scale', type=float, default=0.01)
    p.add_argument('--step_penalty', type=float, default=0.0)
    p.add_argument('--reward_switch_after_steps', type=int, default=0)
    p.add_argument('--cube_reward_mode', type=str, default='dense', choices=cube_reward_choices,
                   help=(
                       'Cube-state reward mode: sparse_final (episode success), '
                       'sparse_intermediate (per cube solved), '
                       'dense (phase-based progress: reach->grasp->carry->place).'
                   ))
    p.add_argument('--cube_success_reward', type=float, default=1.0,
                   help='Sparse success reward used by cube reward modes.')
    p.add_argument('--cube_subgoal_grasp_reward', type=float, default=0.25,
                   help='Sparse grasp-event reward used by cube subgoal modes.')
    p.add_argument('--cube_subgoal_place_reward', type=float, default=1.0,
                   help='Sparse place-event reward used by cube subgoal modes.')
    p.add_argument('--cube_subgoal_drop_penalty', type=float, default=0.0,
                   help='Drop penalty applied on solved-count regressions in subgoal modes (typically <= 0).')
    p.add_argument('--cube_subgoal_grasp_error_threshold', type=float, default=0.08,
                   help='Error threshold used to trigger a one-time grasp event proxy in subgoal modes.')
    p.add_argument('--cube_dense_progress_scale', type=float, default=1.0,
                   help='Scale for dense phase-progress reward increments (reach/carry progress terms).')
    p.add_argument('--cube_dense_progress_clip', type=float, default=0.0,
                   help='Optional max per-step phase progress contribution before scaling (0 disables clipping).')
    p.add_argument(
        '--pvp_proxy_value_bound',
        type=float,
        default=1.0,
        help='PVP only: target proxy Q magnitude used for teacher/student action supervision (+bound / -bound).',
    )
    p.add_argument(
        '--pvp_include_env_reward_in_td',
        action='store_true',
        default=False,
        help='PVP only: include the environment reward in the TD target instead of running reward-free TD.',
    )
    p.add_argument(
        '--eil_threshold',
        type=float,
        default=0.0,
        help='EIL only: scalar Q threshold that separates acceptable from unacceptable actions.',
    )
    p.add_argument(
        '--eil_good_margin',
        type=float,
        default=0.0,
        help='EIL only: margin above --eil_threshold enforced for good/accepted actions.',
    )
    p.add_argument(
        '--eil_bad_margin',
        type=float,
        default=0.01,
        help='EIL only: margin below --eil_threshold enforced for bad/pre-takeover learner actions.',
    )
    p.add_argument(
        '--eil_pair_margin',
        type=float,
        default=0.01,
        help='EIL only: pairwise margin enforcing teacher actions above student proposals on intervened states.',
    )
    p.add_argument(
        '--eil_bad_pre_steps',
        type=int,
        default=8,
        help='EIL only: number of most recent learner-controlled steps before a takeover that are labeled bad.',
    )
    p.add_argument('--switch_env_name', type=str, default=None,
                   help='Optional OGBench env id to switch to after a curriculum step')
    p.add_argument('--switch_env_after_steps', type=int, default=0,
                   help='Global env steps after which to switch to switch_env_name (0 disables)')
    # Intervention / Teacher
    p.add_argument('--use_intervention', action='store_true', default=False)
    p.add_argument('--intervention_mode', type=str, default='agent',
                   choices=['human', 'agent', 'agent_always', 'agent_safety_align', 'agent_safety_progress', 'agent_reward_progress', 'agent_manual_gripper'])
    p.add_argument('--human_input_device', type=str, default='keyboard', choices=['keyboard', 'vr'],
                   help='Human intervention input device. Use vr to consume the persistent raw VR stream.')
    p.add_argument('--human_intervention_threshold', type=float, default=0.1,
                   help='Human intervention action-norm threshold. For VR this is auto-relaxed to ~0 unless explicitly overridden.')
    p.add_argument('--human_intervention_hold_time', type=float, default=0.5,
                   help='Hold time for human intervention takeover logic.')
    p.add_argument('--vr_mode', type=str, default='connect', choices=['connect', 'listen'],
                   help='VR transport mode when --human_input_device vr is active.')
    p.add_argument('--vr_host', type=str, default='',
                   help='VR publisher host when --vr_mode connect. Empty uses the cached endpoint or localhost.')
    p.add_argument('--vr_port', type=int, default=0,
                   help=f'VR transport port (0 uses default {DEFAULT_VR_PORT}).')
    p.add_argument('--vr_cache_path', type=str, default=str(DEFAULT_VR_CACHE_PATH),
                   help='Path to the cached VR endpoint file.')
    p.add_argument('--vr_mapping_path', type=str, default=str(DEFAULT_VR_MAPPING_PATH),
                   help='Path to the saved VR mapping profile.')
    p.add_argument('--vr_reconnect_seconds', type=float, default=2.0,
                   help='Reconnect interval for VR client mode.')
    p.add_argument('--vr_use_saved_mapping', action='store_true', default=True,
                   help='Load the saved VR mapping profile before starting training.')
    p.add_argument('--no_vr_use_saved_mapping', dest='vr_use_saved_mapping', action='store_false',
                   help='Do not load the saved VR mapping profile.')
    p.add_argument('--vr_hand', type=str, default='right', choices=['left', 'right'],
                   help='Fallback VR controller hand if no saved mapping profile is loaded.')
    p.add_argument('--vr_gate_button', type=str, default='grip',
                   help='Fallback VR intervention button if no saved mapping profile is loaded.')
    p.add_argument('--vr_gripper_mirror_toggle_button', type=str, default='none',
                   help='Fallback VR button that toggles gate-off gripper mirroring at runtime if no saved mapping profile is loaded.')
    p.add_argument('--vr_require_gate', action='store_true', default=True,
                   help='Require the configured VR gate button for motion if no saved mapping profile is loaded.')
    p.add_argument('--no_vr_require_gate', dest='vr_require_gate', action='store_false',
                   help='Allow VR motion without a gate button if no saved mapping profile is loaded.')
    p.add_argument('--teacher_type', type=str, default='bfs', choices=['bfs', 'cube_plan', 'cube_markov'])
    p.add_argument('--tolerance_type', type=str, default='angle', choices=['angle', 'l2', 'component'])
    p.add_argument('--tolerance_value', type=float, default=30.0)
    p.add_argument(
        '--tolerance_channel_weights',
        type=str,
        default='',
        help='Optional per-action weights for l2 tolerance. Provide one scalar or comma-separated list matching action dim.',
    )
    p.add_argument(
        '--tolerance_xyz_value',
        type=float,
        default=-1.0,
        help='Component mode: xyz L2 threshold. <=0 falls back to --tolerance_value.',
    )
    p.add_argument(
        '--tolerance_yaw_value',
        type=float,
        default=-1.0,
        help='Component mode: yaw abs threshold. <=0 falls back to --tolerance_value.',
    )
    p.add_argument(
        '--tolerance_gripper_value',
        type=float,
        default=-1.0,
        help='Component mode: gripper abs threshold. <=0 falls back to --tolerance_value.',
    )
    p.add_argument(
        '--tolerance_adaptive_enable',
        action='store_true',
        default=True,
        help='Enable adaptive tightening of component thresholds near target interaction zones.',
    )
    p.add_argument(
        '--no_tolerance_adaptive_enable',
        dest='tolerance_adaptive_enable',
        action='store_false',
        help='Disable adaptive tightening for component thresholds.',
    )
    p.add_argument(
        '--tolerance_adaptive_near_distance',
        type=float,
        default=0.08,
        help='Distance where adaptive threshold scale reaches near-scale.',
    )
    p.add_argument(
        '--tolerance_adaptive_far_distance',
        type=float,
        default=0.30,
        help='Distance where adaptive threshold scale is 1.0.',
    )
    p.add_argument(
        '--tolerance_adaptive_near_scale',
        type=float,
        default=0.35,
        help='Multiplier applied to component thresholds in near zone (0,1].',
    )
    p.add_argument(
        '--binary_gripper_actions',
        action='store_true',
        default=False,
        help='If set, force the final gripper action channel to binary {-1,+1} using --binary_gripper_threshold.',
    )
    p.add_argument(
        '--binary_gripper_threshold',
        type=float,
        default=0.0,
        help='Threshold for binary gripper mapping: final gripper action channel >= threshold -> +1 else -1.',
    )
    p.add_argument(
        '--hard_gripper_intervention',
        action='store_true',
        default=False,
        help='Force teacher takeover on gripper mismatch during critical manipulation phases.',
    )
    p.add_argument(
        '--gripper_intervene_pick_radius',
        type=float,
        default=0.06,
        help='Critical radius (m) around target block where gripper mismatch triggers hard intervention.',
    )
    p.add_argument(
        '--gripper_intervene_place_radius',
        type=float,
        default=0.06,
        help='Critical radius (m) to target placement where gripper mismatch triggers hard intervention.',
    )
    p.add_argument(
        '--gripper_intervene_contact_threshold',
        type=float,
        default=0.3,
        help='Contact threshold above which gripper is considered holding for hard intervention gating.',
    )
    p.add_argument('--hard_block_lethal', action='store_true', default=True)
    p.add_argument('--no_hard_block_lethal', dest='hard_block_lethal', action='store_false')
    p.add_argument('--intervention_enable_after_steps', type=int, default=0)
    p.add_argument('--intervention_safety_margin_frac', type=float, default=0.25,
                   help='Safety margin as a fraction of maze cell size for safety-based intervention modes')
    p.add_argument('--intervention_release_steps', type=int, default=3,
                   help='Consecutive aligned/progressing steps required to release safety intervention')
    p.add_argument('--intervention_reward_patience_steps', type=int, default=5,
                   help='Reward-progress intervention: trigger after this many non-improving steps and release after this many improving steps.')
    p.add_argument('--intervention_reward_improvement_epsilon', type=float, default=1e-6,
                   help='Minimum reward-progress signal increase counted as an improvement.')
    p.add_argument('--intervention_episode_prob', type=float, default=1.0,
                   help='Probability of enabling interventions each episode (1.0 = always)')
    p.add_argument('--intervention_episode_prob_min', type=float, default=0.0,
                   help='Minimum per-episode intervention probability after decay')
    p.add_argument('--intervention_episode_prob_decay_steps', type=int, default=0,
                   help='Per-env steps over which to linearly decay intervention probability (0 disables)')
    p.add_argument('--intervention_episode_prob_decay_start', type=int, default=0,
                   help='Per-env step to start decaying intervention probability')
    p.add_argument('--intervention_episode_prob_seed', type=int, default=None,
                   help='Optional seed for per-episode intervention gating')
    p.add_argument('--teacher_target_mode', type=str, default='sequential', choices=['fixed', 'sequential'],
                   help='Cube-teacher target-block selection mode (sequential is recommended for multi-cube tasks)')
    p.add_argument('--cube_success_tolerance', type=float, default=0.04,
                   help='Success-distance tolerance used to mark per-cube completion for cube teacher target selection')
    # SAC core (trimmed reasonable defaults)
    p.add_argument('--actor_learning_rate', type=float, default=3e-4)
    p.add_argument('--critic_learning_rate', type=float, default=3e-4)
    p.add_argument(
        '--batch_size',
        type=int,
        default=1024,
        help='Total samples per optimizer update (not divided by num_envs).',
    )
    p.add_argument('--buffer_size', type=int, default=1_000_000)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--tau', type=float, default=0.01)
    p.add_argument('--policy_frequency', type=int, default=1)
    p.add_argument('--num_updates', type=int, default=1)
    p.add_argument(
        '--cta_ratio',
        type=int,
        default=1,
        help='Critic-to-actor update ratio. 1 means update actor every critic step, 2 every second critic step.',
    )
    p.add_argument('--learning_starts', type=int, default=10_000)
    p.add_argument('--max_grad_norm', type=float, default=10.0)
    p.add_argument('--init_scale', type=float, default=0.01)
    p.add_argument('--actor_hidden_dim', type=int, default=512)
    p.add_argument('--critic_hidden_dim', type=int, default=1024)
    p.add_argument(
        '--use_layer_norm',
        action='store_true',
        default=False,
        help='Apply LayerNorm after hidden Linear layers in FastSAC MLP backbone/heads.',
    )
    p.add_argument(
        '--layer_norm_eps',
        type=float,
        default=1e-5,
        help='Epsilon used by LayerNorm when --use_layer_norm is enabled.',
    )

    p.add_argument('--arch_shared_trunk', action='store_true', default=False,
                   help='Share an observation trunk between actor and critic(s)')
    p.add_argument('--shared_hidden_dim', type=int, default=512,
                   help='Hidden size for shared trunk when enabled')
    p.add_argument('--num_critics', type=int, default=2,
                   help='Number of critic heads in the ensemble (>=2 recommended)')
    p.add_argument('--obs_mode', type=str, default='state', choices=['state', 'pixels'],
                   help='Observation mode: vector state or pixel images')
    p.add_argument('--pixel_width', type=int, default=64,
                   help='Pixel observation width when obs_mode=pixels')
    p.add_argument('--pixel_height', type=int, default=64,
                   help='Pixel observation height when obs_mode=pixels')
    p.add_argument('--pixel_camera', type=str, default=None,
                   help='Optional MuJoCo camera name for pixel observations (defaults to env setting)')
    p.add_argument('--pixel_camera_mode', type=str, default='global',
                   choices=['global', 'agent_local', 'first_person'],
                   help='Camera behaviour when using pixel observations without an explicit camera_name')
    p.add_argument('--pixel_local_view_size', type=float, default=12.0,
                   help="World-space width/height (in meters) of the local bird's-eye crop")
    p.add_argument('--pixel_local_camera_height', type=float, default=None,
                   help='Optional override for camera height when using agent_local mode (defaults to derived value)')
    p.add_argument('--pixel_first_person_distance', type=float, default=3.0,
                   help='Distance between camera and lookat point for first_person mode')
    p.add_argument('--pixel_first_person_height', type=float, default=1.0,
                   help='Height offset applied to the camera and lookat point for first_person mode')
    p.add_argument('--pixel_first_person_lookahead', type=float, default=2.0,
                   help='Forward offset (in meters) added to the lookat point for first_person mode')
    p.add_argument('--pixel_first_person_pitch', type=float, default=-15.0,
                   help='Camera pitch (degrees) applied in first_person mode (negative looks down)')
    p.add_argument('--decouple_view', action='store_true', default=False,
                   help='Decouple movement and view direction (adds a view-delta action)')
    p.add_argument('--view_delta_scale', type=float, default=3.14159265,
                   help='Max radians applied to view delta when decouple_view is enabled')
    p.add_argument('--goal_relative_history', action='store_true', default=False,
                   help='Append agent-centric goal deltas to the non-visual history trunk')
    p.add_argument('--goal_relative_scale', type=float, default=10.0,
                   help='Meters mapped to full-scale goal encoding (used with goal_relative_history)')
    p.add_argument('--pixel_conv_channels', type=str, default='32,64,64',
                   help='Comma-separated Conv2d channel sizes for the pixel backbone')
    p.add_argument('--pixel_kernel_sizes', type=str, default='8,4,3',
                   help='Comma-separated kernel sizes for each conv layer (defaults to 8,4,3)')
    p.add_argument('--pixel_strides', type=str, default='4,2,1',
                   help='Comma-separated strides for each conv layer (defaults to 4,2,1)')
    p.add_argument('--pixel_final_pool', type=int, default=0,
                   help='If >0, apply AdaptiveAvgPool2d to this spatial size after conv stack')
    p.add_argument('--pixel_random_shift_pad', type=int, default=4,
                   help='Pad size for DrQ-style random shifts (set to 0 to disable)')
    p.add_argument('--alpha_min', type=float, default=0.0,
                   help='Minimum entropy temperature (alpha). Set to 0 to disable lower clamp.')
    p.add_argument('--alpha_max', type=float, default=1.0,
                   help='Maximum entropy temperature (alpha).')
    p.add_argument('--alpha_init', type=float, default=1e-3,
                   help='Initial entropy temperature (alpha) before any updates.')
    p.add_argument('--fixed_alpha', type=float, default=-1.0,
                   help='If >= 0, hold alpha fixed at this exact value and disable alpha updates.')
    p.add_argument(
        '--alpha_update_student_only',
        action='store_true',
        default=False,
        help='When replay rows carry teacher_intervened markers, update alpha using only non-intervened rows.',
    )
    p.add_argument('--alpha_freeze_steps', type=int, default=0,
                   help='Disable alpha updates until this many env steps have elapsed (0 disables).')
    p.add_argument('--debug_pixel_dump', action='store_true', default=False,
                   help='Log raw vs normalized observation stats and sample actions at first step')
    p.add_argument('--store_denied_actions', action='store_true', default=False,
                   help='Add denied student actions to replay buffer with penalty reward')
    p.add_argument('--denied_action_penalty', type=float, default=-1.0,
                   help='Reward assigned to denied student actions when stored')
    # Demo prefill (teacher-generated data)
    p.add_argument('--demo_prefill_steps', type=int, default=0,
                   help='Number of env steps to prefill replay buffer with teacher demos (0 disables)')
    p.add_argument('--demo_prefill_episodes', type=int, default=0,
                   help='Number of demo episodes to prefill (0 disables, overrides steps)')
    p.add_argument('--demo_prefill_num_envs', type=int, default=0,
                   help='Number of vector envs to use during demo prefill only (0 reuses --num_envs)')
    p.add_argument('--demo_prefill_intervention_mode', type=str, default='agent_safety_progress',
                   choices=['human', 'agent', 'agent_always', 'agent_safety_align', 'agent_safety_progress', 'agent_reward_progress', 'agent_manual_gripper'],
                   help='Intervention mode to use during demo prefill')
    p.add_argument('--demo_prefill_enable_after_steps', type=int, default=0,
                   help='Warm-up steps per env before demo interventions engage')
    p.add_argument('--demo_prefill_episode_prob', type=float, default=1.0,
                   help='Per-episode intervention gate probability during demo prefill')
    p.add_argument('--demo_prefill_hard_block_lethal', action='store_true', default=False,
                   help='Enable hard blocking of lethal moves during demo prefill')
    p.add_argument('--demo_prefill_target', type=str, default='replay',
                   choices=['replay', 'demo'],
                   help='Which buffer to prefill with teacher demos')
    p.add_argument('--demo_buffer_enable', action='store_true', default=False,
                   help='Enable a separate demo replay buffer for mixed sampling')
    p.add_argument('--demo_buffer_capacity', type=int, default=200000,
                   help='Capacity of the demo replay buffer')
    p.add_argument('--demo_sample_ratio', type=float, default=0.0,
                   help='Fraction of each update batch sampled from demo buffer (0..1)')
    p.add_argument('--store_intervened_in_demo_buffer', action='store_true', default=False,
                   help='When online interventions occur, also copy those executed transitions into the demo buffer (requires --demo_buffer_enable).')
    p.add_argument('--demo_dataset_path', type=str, default='',
                   help='Optional offline manipulation transition dataset (.npz) to load into replay/demo before training.')
    p.add_argument('--demo_dataset_dir', type=str, default='',
                   help='Dataset search root for --demo_dataset_auto_load (defaults to the shared manipulation dataset dir).')
    p.add_argument('--demo_dataset_auto_load', action='store_true', default=False,
                   help='Automatically load the latest matching manipulation dataset for env_name before training.')
    p.add_argument('--demo_dataset_target', type=str, default='demo', choices=['demo', 'replay'],
                   help='Target buffer for offline manipulation dataset loading.')
    p.add_argument('--demo_dataset_max_rows', type=int, default=0,
                   help='Optional max number of rows to load from the offline manipulation dataset (0 = all).')
    p.add_argument('--export_replay_dataset_interval', type=int, default=0,
                   help='Env-step interval for atomic replay-dataset snapshots during manipulation training (0 disables).')
    p.add_argument('--export_replay_dataset_path', type=str, default='',
                   help='Optional fixed .npz path for periodic/final manipulation replay snapshots.')
    p.add_argument('--export_replay_dataset_dir', type=str, default='',
                   help='Directory root for auto-named manipulation replay snapshots.')
    p.add_argument('--export_replay_dataset_label', type=str, default='online_replay',
                   help='Label used when auto-naming manipulation replay snapshots.')
    p.add_argument('--export_replay_dataset_max_rows', type=int, default=0,
                   help='Optional max rows to export per manipulation replay snapshot (0 = all).')
    # Logging
    p.add_argument('--use_wandb', action='store_true', default=False)
    p.add_argument('--project', type=str, default='ogbench-rsl-rl')
    p.add_argument('--exp_name', type=str, default=None)
    p.add_argument('--save_interval', type=int, default=200000,
                   help='Env-step interval for checkpoint saves (0 disables)')
    p.add_argument('--log_interval', type=int, default=200)
    p.add_argument(
        '--eval_render_mode',
        type=str,
        default='none',
        choices=['none', 'human'],
        help='Optional live rendering during periodic training eval (human requires eval_num_envs=1).',
    )
    p.add_argument(
        '--allow_simultaneous_train_eval_render',
        action='store_true',
        default=False,
        help=(
            'Allow train_render_mode=human and eval_render_mode=human at the same time. '
            'Disabled by default to avoid dual-viewer instability/segfaults.'
        ),
    )
    p.add_argument('--profile_timing', action='store_true', default=False,
                   help='Log per-step timing breakdown (ms + %%): action/env/info/replay-sample/update/misc')
    p.add_argument('--compute_q_diagnostics', action='store_true', default=False,
                   help='Compute per-step critic disagreement/Q-min diagnostics (adds extra critic forward each env step).')
    p.add_argument('--post_switch_viz_multiplier', type=int, default=1,
                   help='Reduce checkpoint/viz interval by this factor after curriculum/env switch (>=1)')
    p.add_argument('--disagreement_hist_edges', type=str, default='',
                   help='Comma-separated positive edges for disagreement hist bins (teacher/non-teacher). Empty to disable histogram logging.')
    p.add_argument('--disagreement_thresholds', type=str, default='0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0,2.0,5.0',
                   help='Comma-separated thresholds to log fraction of interventions with disagreement >= threshold')
    # Misc
    p.add_argument('--compile', action='store_true', default=False)
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--amp_dtype', type=str, default='bf16', choices=['bf16','fp16'])
    # Preference buffer (pairwise ranking on intervened rows: teacher vs student)
    p.add_argument('--pref_buffer_enable', action='store_true', default=False,
                   help='Enable preference buffer storing pairs (s, a_teacher, a_student) and ranking loss')
    p.add_argument('--pref_capacity', type=int, default=100000,
                   help='Capacity of preference buffer (pairs)')
    p.add_argument('--pref_sample_ratio', type=float, default=0.5,
                   help='Fraction of update batch for preference pairs (0..1)')
    p.add_argument(
        '--pref_sampling_mode',
        type=str,
        default='independent',
        choices=['independent', 'linked'],
        help=(
            "Preference sample source: 'independent' uses PreferencePairBuffer sampling; "
            "'linked' derives preference pairs directly from sampled replay transitions "
            "(teacher action = executed action, student action = stored student proposal)."
        ),
    )
    p.add_argument(
        '--pref_linked_action_epsilon',
        type=float,
        default=1e-6,
        help=(
            "Intervention detection threshold for linked preference sampling. "
            "A replay row is treated as intervention-linked when ||a_exec - a_student||_1 > epsilon."
        ),
    )
    p.add_argument('--pref_rank_weight', type=float, default=1.0,
                   help='Weight for the pairwise ranking loss added to critic loss')
    p.add_argument('--pref_rank_margin', type=float, default=0.1,
                   help='Margin for ranking loss: softplus(margin - (Qpos - Qneg))')
    p.add_argument(
        '--pref_critic_scope',
        type=str,
        default='all',
        choices=['all', 'min'],
        help='Apply preference loss to every critic head or only to the min-Q teacher/student pair.',
    )
    p.add_argument(
        '--pref_loss_type',
        type=str,
        default='margin',
        choices=['margin', 'hinge', 'bradley_terry', 'lagrangian'],
        help='Preference loss formulation for critic updates.',
    )
    p.add_argument(
        '--pref_lambda_init',
        type=float,
        default=0.0,
        help='Initial Lagrange multiplier value used when --pref_loss_type=lagrangian.',
    )
    p.add_argument(
        '--pref_lambda_lr',
        type=float,
        default=1e-3,
        help='Dual ascent step size for lagrangian preference optimization.',
    )
    p.add_argument(
        '--pref_lambda_max',
        type=float,
        default=10.0,
        help='Upper clip for Lagrange multiplier (<=0 disables clipping).',
    )
    p.add_argument(
        '--pref_lambda_ema',
        type=float,
        default=0.9,
        help='EMA factor for preference violation in dual update (0 disables EMA).',
    )
    p.add_argument(
        '--pref_violation_clip',
        type=float,
        default=10.0,
        help='Clip positive violation before lagrangian loss (<=0 disables clipping).',
    )
    p.add_argument(
        '--pref_violation_target',
        type=float,
        default=0.0,
        help='Target violation level for lagrangian dual update: lambda += lr * (violation_estimate - target).',
    )
    p.add_argument(
        '--pref_lagrangian_violation_type',
        type=str,
        default='hinge',
        choices=['hinge', 'smooth'],
        help='Violation shape used when --pref_loss_type=lagrangian.',
    )
    p.add_argument(
        '--pref_stopgrad_positive',
        action='store_true',
        default=False,
        help='Detach positive preference Q term so ranking loss only pushes down the negative sample.',
    )
    # Supervised actor regularization (DAgger-style BC on teacher actions)
    p.add_argument('--actor_bc_weight_demo', type=float, default=0.0,
                   help='Weight for actor BC loss on demo-buffer actions (0 disables)')
    p.add_argument('--actor_bc_weight_pref', type=float, default=0.0,
                   help='Weight for actor BC loss on teacher actions sampled from preference buffer (0 disables)')
    # Replay buffer reset on curriculum switch
    p.add_argument('--reset_replay_on_switch', action='store_true', default=False,
                   help='Reset main replay buffer when reward_switch_after_steps is reached')
    p.add_argument('--reset_critic_on_switch', action='store_true', default=False,
                   help='Reload critic weights/optimiser to initial state at curriculum switch')
    # Policy visualization
    p.add_argument('--viz_on_checkpoint', action='store_true', default=False,
                   help='Render policy/critic maps whenever a checkpoint is saved')
    p.add_argument('--viz_grid_resolution', type=int, default=32,
                   help='Grid resolution for policy maps if enabled')
    p.add_argument('--viz_quiver_stride', type=int, default=2,
                   help='Stride for quiver arrows in policy maps')
    p.add_argument('--viz_device', type=str, default='cpu',
                   help='Device to use when generating policy maps')
    p.add_argument('--viz_seed', type=int, default=0,
                   help='Seed used to sample/lock the visualization goal location')
    p.add_argument('--viz_first_step', type=int, default=None,
                   help='Force the first checkpoint/viz at this env-step (even if save_interval is larger)')
    return p

def build_eval_parser() -> argparse.ArgumentParser:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Interactive evaluation of trained RSL-RL agents')
    cube_reward_choices = [
        'sparse_final',
        'sparse_intermediate',
        'dense',
    ]
    
    # Model and environment
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the trained model (.pt file). Optional when using --controller random/human.')
    parser.add_argument('--env_name', type=str, default='pointmaze-medium-v0',
                        help='OGBench environment name')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run on (auto, cpu, cuda)')
    parser.add_argument('--policy_type', type=str, default='auto',
                        choices=['auto', 'rsl-rl', 'fastsac', 'fastsac_v2', 'drqv2'],
                        help='Policy checkpoint format to load')
    parser.add_argument('--controller', type=str, default='policy',
                        choices=['policy', 'random', 'human', 'keyboard'],
                        help='Source of actions: trained policy, random actions, human-only, or keyboard control')
    
    # Visualization
    parser.add_argument('--render_mode', type=str, default='human',
                        choices=['human', 'rgb_array'],
                        help='Rendering mode')
    parser.add_argument('--width', type=int, default=800,
                        help='Render width')
    parser.add_argument('--height', type=int, default=600,
                        help='Render height')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target FPS for rendering')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Enable verbose evaluation logging')
    parser.add_argument('--log_q_values', action='store_true', default=False,
                        help='Log critic Q values for performed actions during evaluation')
    parser.add_argument('--log_q_every', type=int, default=50,
                        help='Step interval for logging Q values during evaluation')
    parser.add_argument('--show_q_overlay', dest='show_q_overlay', action='store_true', default=True,
                        help='Overlay critic Q values in the interactive renderer')
    parser.add_argument('--no_show_q_overlay', dest='show_q_overlay', action='store_false',
                        help='Disable Q-value overlay')
    
    # Observation configuration (match training)
    parser.add_argument('--obs_mode', type=str, default=None, choices=['state', 'pixels'],
                        help='Observation mode override (defaults to checkpoint metadata)')
    parser.add_argument('--include_goal', dest='include_goal', action='store_true', default=True,
                        help='Include goal coordinates in observations')
    parser.add_argument('--no_include_goal', dest='include_goal', action='store_false',
                        help='Exclude goal coordinates from observations')
    parser.add_argument('--include_distance', action='store_true', default=False,
                        help='Include distance to goal in observations')
    parser.add_argument('--include_direction', action='store_true', default=False,
                        help='Include direction to goal in observations')
    parser.add_argument('--include_velocity', action='store_true', default=False,
                        help='Include velocity features in observations')
    parser.add_argument('--goal_marker_color', type=str, default='auto',
                        choices=['auto', 'red', 'green', 'blue'],
                        help='Override maze goal marker color (auto keeps env default)')
    parser.add_argument('--frame_stack', type=int, default=None,
                        help='Number of stacked frames expected by the policy (defaults to checkpoint metadata)')
    parser.add_argument('--use_local_actions', action='store_true', default=False,
                        help='Interpret actions in the agent-local frame (auto-filled from checkpoint if available)')
    parser.add_argument('--no_use_local_actions', dest='use_local_actions', action='store_false',
                        help='Force global actions even if checkpoint requested local frame')
    parser.add_argument('--se2_translation_scale', type=float, default=0.2,
                        help='Scale factor for SE(2) latent warps when local actions are enabled')
    parser.add_argument('--goal_relative_scale', type=float, default=10.0,
                        help='Scale factor for agent-centric goal deltas (used when goal history is enabled)')

    # Pixel observation overrides (auto-filled from checkpoint if available)
    parser.add_argument('--pixel_width', type=int, default=None,
                        help='Pixel observation width (defaults to training checkpoint)')
    parser.add_argument('--pixel_height', type=int, default=None,
                        help='Pixel observation height (defaults to training checkpoint)')
    parser.add_argument('--pixel_camera', type=str, default=None,
                        help='Camera name for pixel observations (defaults to training checkpoint)')
    parser.add_argument('--pixel_camera_mode', type=str, default='global',
                        choices=['global', 'agent_local', 'first_person'],
                        help='Camera behaviour when relying on the dynamic MuJoCo free camera')
    parser.add_argument('--pixel_local_view_size', type=float, default=12.0,
                        help="World-space width/height (in meters) captured by the local bird's-eye view")
    parser.add_argument('--pixel_local_camera_height', type=float, default=None,
                        help='Override camera height for agent_local mode (default derives from view size)')
    parser.add_argument('--pixel_first_person_distance', type=float, default=3.0,
                        help='Distance between camera and lookat point for first_person mode')
    parser.add_argument('--pixel_first_person_height', type=float, default=1.0,
                        help='Height offset applied to the camera and lookat point for first_person mode')
    parser.add_argument('--pixel_first_person_lookahead', type=float, default=2.0,
                        help='Forward offset (meters) for the lookat point in first_person mode')
    parser.add_argument('--pixel_first_person_pitch', type=float, default=-15.0,
                        help='Camera pitch (degrees, negative looks down) for first_person mode')
    parser.add_argument('--decouple_view', action='store_true', default=False,
                        help='Decouple movement and view direction (adds a view-delta action)')
    parser.add_argument('--view_delta_scale', type=float, default=3.14159265,
                        help='Max radians applied to view delta when decouple_view is enabled')
    parser.add_argument('--mirror_human_render', dest='mirror_human_render', action='store_true', default=False,
                        help='Mirror rgb_array rollouts to a separate human-rendered window')
    parser.add_argument('--no_mirror_human_render', dest='mirror_human_render', action='store_false',
                        help='Disable mirrored human-render window (useful on headless machines)')
    parser.add_argument('--policy_mujoco_gl', type=str, default='auto',
                        choices=['auto', 'egl', 'glfw'],
                        help='Backend for policy environment (auto uses egl unless mirror rendering requires glfw)')

    # Reward shaping (should mirror training wrapper settings)
    parser.add_argument('--reward_type', type=str, default='sparse',
                        choices=['sparse', 'dense', 'combined', 'none'],
                        help='Reward type for DetailedRewardWrapper')
    parser.add_argument('--dense_reward_scale', type=float, default=0.01,
                        help='Scale for dense reward shaping (if applicable)')
    parser.add_argument('--step_penalty', type=float, default=0.0,
                        help='Per-step penalty applied by DetailedRewardWrapper')
    parser.add_argument('--reward_switch_after_steps', type=int, default=0,
                        help='Switch reward to sparse after this many steps (curriculum)')
    parser.add_argument('--cube_reward_mode', type=str, default='dense', choices=cube_reward_choices,
                        help=(
                            'Cube-state reward mode: sparse_final (episode success), '
                            'sparse_intermediate (per cube solved), '
                            'dense (phase-based progress: reach->grasp->carry->place).'
                        ))
    parser.add_argument('--cube_success_reward', type=float, default=1.0,
                        help='Sparse success reward used by cube reward modes.')
    parser.add_argument('--cube_subgoal_grasp_reward', type=float, default=0.25,
                        help='Sparse grasp-event reward used by cube subgoal modes.')
    parser.add_argument('--cube_subgoal_place_reward', type=float, default=1.0,
                        help='Sparse place-event reward used by cube subgoal modes.')
    parser.add_argument('--cube_subgoal_drop_penalty', type=float, default=0.0,
                        help='Drop penalty applied on solved-count regressions in subgoal modes (typically <= 0).')
    parser.add_argument('--cube_subgoal_grasp_error_threshold', type=float, default=0.08,
                        help='Error threshold used to trigger a one-time grasp event proxy in subgoal modes.')
    parser.add_argument('--cube_dense_progress_scale', type=float, default=1.0,
                        help='Scale for dense phase-progress reward increments (reach/carry progress terms).')
    parser.add_argument('--cube_dense_progress_clip', type=float, default=0.0,
                        help='Optional max per-step phase progress contribution before scaling (0 disables clipping).')

    # Evaluation
    parser.add_argument('--max_episode_steps', type=int, default=500,
                        help='Maximum steps per episode')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of episodes to run (0 = infinite)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--print_interventions', action='store_true', default=True,
                        help='Print when the teacher intervenes and why')
    parser.add_argument('--success_distance_epsilon', type=float, default=0.05,
                        help='Distance threshold (in env units) to count a terminal state as goal reached')

    # Intervention / Teleop
    parser.add_argument('--intervention_mode', type=str, default='none',
                        choices=['none', 'human', 'agent', 'agent_always', 'agent_safety_align', 'agent_safety_progress', 'agent_reward_progress', 'agent_manual_gripper'],
                        help='Intervention mode: none, human teleop, or agent teacher')
    parser.add_argument('--teacher_type', type=str, default='bfs', choices=['bfs', 'cube_plan', 'cube_markov'],
                        help='Teacher type when intervention_mode=agent')
    parser.add_argument('--tolerance_type', type=str, default='angle', choices=['angle', 'l2'],
                        help='Intervention tolerance metric (agent mode)')
    parser.add_argument('--tolerance_value', type=float, default=30.0,
                        help='Tolerance threshold (deg for angle; abs for l2)')
    parser.add_argument(
        '--tolerance_channel_weights',
        type=str,
        default='',
        help='Optional per-action weights for l2 tolerance. Provide one scalar or comma-separated list matching action dim.',
    )
    parser.add_argument('--hard_block_lethal', action='store_true', default=True,
                        help='Intervene if student would step into lethal cell')
    parser.add_argument('--no_hard_block_lethal', dest='hard_block_lethal', action='store_false')
    parser.add_argument('--intervention_enable_after_steps', type=int, default=0,
                        help='Warm-up steps before agent teacher interventions engage')
    parser.add_argument('--intervention_safety_margin_frac', type=float, default=0.25,
                        help='Safety margin as a fraction of maze cell size for safety-based intervention modes')
    parser.add_argument('--intervention_release_steps', type=int, default=3,
                        help='Consecutive aligned/progressing steps required to release safety intervention')
    parser.add_argument('--intervention_reward_patience_steps', type=int, default=5,
                        help='Reward-progress intervention: trigger after this many non-improving steps and release after this many improving steps.')
    parser.add_argument('--intervention_reward_improvement_epsilon', type=float, default=1e-6,
                        help='Minimum reward-progress signal increase counted as an improvement.')
    
    # Action processing (should match training settings)
    parser.add_argument('--action_scale', type=float, default=1.0,
                        help='Action scaling factor (should match training)')
    parser.add_argument('--clip_actions', action='store_true', default=True,
                        help='Clip actions to [-1, 1] (should match training)')
    parser.add_argument('--headless', action='store_true', default=False,
                        help='Run evaluation without pygame windows (forces rgb_array render)')
    
    return parser
