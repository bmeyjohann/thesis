from __future__ import annotations

import argparse

def build_train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # Env
    p.add_argument('--env_name', type=str, default='pointmaze-arena-danger-lethal-v0')
    p.add_argument('--num_envs', type=int, default=64)
    p.add_argument('--total_timesteps', type=int, default=1_000_000)
    p.add_argument('--device', type=str, default='auto')
    # Observations
    p.add_argument('--include_goal', action='store_true', default=True)
    p.add_argument('--include_distance', action='store_true', default=False)
    p.add_argument('--include_direction', action='store_true', default=False)
    p.add_argument('--include_velocity', action='store_true', default=False)
    p.add_argument('--goal_marker_color', type=str, default='auto',
                   choices=['auto', 'red', 'green', 'blue'],
                   help='Override maze goal marker color (auto keeps env default)')
    # Rewards
    p.add_argument('--reward_type', type=str, default='sparse', choices=['sparse','dense','combined'])
    p.add_argument('--dense_reward_scale', type=float, default=0.01)
    p.add_argument('--step_penalty', type=float, default=0.0)
    p.add_argument('--reward_switch_after_steps', type=int, default=0)
    p.add_argument('--switch_env_name', type=str, default=None,
                   help='Optional OGBench env id to switch to after a curriculum step')
    p.add_argument('--switch_env_after_steps', type=int, default=0,
                   help='Global env steps after which to switch to switch_env_name (0 disables)')
    # Intervention / Teacher
    p.add_argument('--use_intervention', action='store_true', default=False)
    p.add_argument('--intervention_mode', type=str, default='agent', choices=['human','agent'])
    p.add_argument('--teacher_type', type=str, default='bfs', choices=['bfs'])
    p.add_argument('--tolerance_type', type=str, default='angle', choices=['angle','l2'])
    p.add_argument('--tolerance_value', type=float, default=30.0)
    p.add_argument('--hard_block_lethal', action='store_true', default=True)
    p.add_argument('--no_hard_block_lethal', dest='hard_block_lethal', action='store_false')
    p.add_argument('--intervention_enable_after_steps', type=int, default=0)
    # SAC core (trimmed reasonable defaults)
    p.add_argument('--actor_learning_rate', type=float, default=3e-4)
    p.add_argument('--critic_learning_rate', type=float, default=3e-4)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--buffer_size', type=int, default=1_000_000)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--tau', type=float, default=0.01)
    p.add_argument('--policy_frequency', type=int, default=1)
    p.add_argument('--num_updates', type=int, default=1)
    p.add_argument('--learning_starts', type=int, default=10_000)
    p.add_argument('--max_grad_norm', type=float, default=10.0)
    p.add_argument('--init_scale', type=float, default=0.01)
    p.add_argument('--actor_hidden_dim', type=int, default=512)
    p.add_argument('--critic_hidden_dim', type=int, default=1024)

    p.add_argument('--arch_shared_trunk', action='store_true', default=False,
                   help='Share an observation trunk between actor and critic(s)')
    p.add_argument('--shared_hidden_dim', type=int, default=512,
                   help='Hidden size for shared trunk when enabled')
    p.add_argument('--num_critics', type=int, default=2,
                   help='Number of critic heads (2 or 3 supported)')
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
    p.add_argument('--alpha_freeze_steps', type=int, default=0,
                   help='Disable alpha updates until this many env steps have elapsed (0 disables).')
    p.add_argument('--debug_pixel_dump', action='store_true', default=False,
                   help='Log raw vs normalized observation stats and sample actions at first step')
    p.add_argument('--store_denied_actions', action='store_true', default=False,
                   help='Add denied student actions to replay buffer with penalty reward')
    p.add_argument('--denied_action_penalty', type=float, default=-1.0,
                   help='Reward assigned to denied student actions when stored')
    # Intervention reward shaping variants
    p.add_argument('--intervention_reward_mode', type=str, default='none',
                   choices=['none', 'penalty_student', 'bonus_teacher'],
                   help='How to shape reward/actions on intervention')
    p.add_argument('--intervention_reward_value', type=float, default=0.0,
                   help='Magnitude for intervention reward shaping (e.g., 0.1)')
    p.add_argument('--bonus_teacher_value', type=float, default=0.0,
                   help='Additional reward added when teacher action is applied (can combine with penalty modes)')
    # Logging
    p.add_argument('--use_wandb', action='store_true', default=False)
    p.add_argument('--project', type=str, default='ogbench-rsl-rl')
    p.add_argument('--exp_name', type=str, default=None)
    p.add_argument('--save_interval', type=int, default=200000,
                   help='Env-step interval for checkpoint saves (0 disables)')
    p.add_argument('--log_interval', type=int, default=200)
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
    # Counterfactual buffer (student-denied actions) for fast adaptation
    p.add_argument('--cf_buffer_enable', action='store_true', default=False,
                   help='Enable counterfactual buffer for denied student actions')
    p.add_argument('--cf_capacity', type=int, default=100000,
                   help='Capacity of CF buffer (rows)')
    p.add_argument('--cf_sample_ratio', type=float, default=0.5,
                   help='Fraction of batch for CF critic loss (0..1)')
    p.add_argument('--cf_penalty', type=float, default=1.0,
                   help='Positive penalty magnitude; critic target becomes -abs(value) for denied actions')
    p.add_argument('--cf_q_weight', type=float, default=1.0,
                   help='Weight for CF critic penalty loss')
    # Preference buffer (pairwise ranking on intervened rows: teacher vs student)
    p.add_argument('--pref_buffer_enable', action='store_true', default=False,
                   help='Enable preference buffer storing pairs (s, a_teacher, a_student) and ranking loss')
    p.add_argument('--pref_capacity', type=int, default=100000,
                   help='Capacity of preference buffer (pairs)')
    p.add_argument('--pref_sample_ratio', type=float, default=0.5,
                   help='Fraction of update batch for preference pairs (0..1)')
    p.add_argument('--pref_rank_weight', type=float, default=1.0,
                   help='Weight for the pairwise ranking loss added to critic loss')
    p.add_argument('--pref_rank_margin', type=float, default=0.1,
                   help='Margin for ranking loss: softplus(margin - (Qpos - Qneg))')
    # Preference-TD buffer (balanced TD samples: teacher real transition; student synthetic terminal negative)
    p.add_argument('--pref_td_buffer_enable', action='store_true', default=False,
                   help='Enable preference-TD buffer that keeps balanced teacher/student TD transitions')
    p.add_argument('--pref_td_capacity', type=int, default=100000,
                   help='Capacity per-role (teacher/student) for preference-TD buffer')
    p.add_argument('--pref_td_sample_ratio', type=float, default=0.5,
                   help='Fraction of update batch to draw from preference-TD buffer (split 50/50 teacher/student)')
    p.add_argument('--pref_td_q_weight', type=float, default=1.0,
                   help='Weight for additional critic TD loss from preference-TD samples')
    p.add_argument('--pref_td_penalty_value', type=float, default=0.1,
                   help='Negative reward assigned to synthetic student terminal in preference-TD buffer')
    p.add_argument('--pref_td_teacher_bonus_value', type=float, default=0.0,
                   help='Optional extra reward added to teacher transitions in preference-TD buffer')
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
                        choices=['policy', 'random', 'human'],
                        help='Source of actions: trained policy, random actions, or human-only (zero-action baseline)')
    
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
    parser.add_argument('--mirror_human_render', dest='mirror_human_render', action='store_true', default=False,
                        help='Mirror rgb_array rollouts to a separate human-rendered window')
    parser.add_argument('--no_mirror_human_render', dest='mirror_human_render', action='store_false',
                        help='Disable mirrored human-render window (useful on headless machines)')
    parser.add_argument('--policy_mujoco_gl', type=str, default='auto',
                        choices=['auto', 'egl', 'glfw'],
                        help='Backend for policy environment (auto uses egl unless mirror rendering requires glfw)')

    # Reward shaping (should mirror training wrapper settings)
    parser.add_argument('--reward_type', type=str, default='sparse',
                        choices=['sparse', 'dense', 'combined'],
                        help='Reward type for DetailedRewardWrapper')
    parser.add_argument('--dense_reward_scale', type=float, default=0.01,
                        help='Scale for dense reward shaping (if applicable)')
    parser.add_argument('--step_penalty', type=float, default=0.0,
                        help='Per-step penalty applied by DetailedRewardWrapper')
    parser.add_argument('--reward_switch_after_steps', type=int, default=0,
                        help='Switch reward to sparse after this many steps (curriculum)')

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
                        choices=['none', 'human', 'agent'],
                        help='Intervention mode: none, human teleop, or agent teacher')
    parser.add_argument('--teacher_type', type=str, default='bfs', choices=['bfs'],
                        help='Teacher type when intervention_mode=agent')
    parser.add_argument('--tolerance_type', type=str, default='angle', choices=['angle', 'l2'],
                        help='Intervention tolerance metric (agent mode)')
    parser.add_argument('--tolerance_value', type=float, default=30.0,
                        help='Tolerance threshold (deg for angle; abs for l2)')
    parser.add_argument('--hard_block_lethal', action='store_true', default=True,
                        help='Intervene if student would step into lethal cell')
    parser.add_argument('--no_hard_block_lethal', dest='hard_block_lethal', action='store_false')
    parser.add_argument('--intervention_enable_after_steps', type=int, default=0,
                        help='Warm-up steps before agent teacher interventions engage')
    
    # Action processing (should match training settings)
    parser.add_argument('--action_scale', type=float, default=1.0,
                        help='Action scaling factor (should match training)')
    parser.add_argument('--clip_actions', action='store_true', default=True,
                        help='Clip actions to [-1, 1] (should match training)')
    parser.add_argument('--headless', action='store_true', default=False,
                        help='Run evaluation without pygame windows (forces rgb_array render)')
    
    return parser
