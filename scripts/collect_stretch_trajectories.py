#!/usr/bin/env python3
"""
Collect action-labeled trajectories from a Stretch MuJoCo (or Gym-like) environment
and save them as a pickle list of trajectory dicts compatible with the repo's
`DictToMDPPathLoader` expectations (keys: observations, next_observations,
actions, rewards, terminals, agent_infos, env_infos).

Usage examples:
  # Use a gym id:
  python scripts/collect_stretch_trajectories.py --env-id StretchMuJoCo-v0 --num-episodes 50 --horizon 400 --save-path data/stretch_demos.pkl --obs-dict

  # Use a python importable env factory: module:Factory (must be callable)
  python scripts/collect_stretch_trajectories.py --env-factory my_stretch_envs:make_stretch_env --num-episodes 20 --save-path data/stretch_demos.pkl --obs-dict

Notes / assumptions:
 - The script expects a Gym-like API: reset() -> obs, step(action) -> (next_obs, reward, done, info)
 - If `--obs-dict` is passed we will try to populate each observation entry with an
   'image_observation' field (the loader expects this key when obs_dict=True). If the
   env's observation is already a dict and contains 'image_observation' it will be used.
   Otherwise, the script will call env.render(mode='rgb_array') to obtain an image.
 - `agent_infos` and `env_infos` are saved as empty dicts per time step by default. You
   can extend the script to populate them if your env provides diagnostics.
 - Because Stretch setups vary, pass an env factory or adapt the script to your local
   Stretch MuJoCo wrapper.

"""
import argparse
import importlib
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

# Add parent directory to path so we can import scripts module
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _to_cwh_image(img, input_format):
    """
    Convert image to CWH layout (channels, width, height).

    Supported input formats:
      - CHW: (C, H, W) or flattened CHW
      - CWH: (C, W, H) or flattened CWH
      - HWC: (H, W, C) or flattened HWC
    """
    arr = np.asarray(img)

    if arr.ndim == 1:
        # Assume square RGB image for flattened inputs.
        side = int(round(np.sqrt(arr.size / 3.0)))
        if 3 * side * side != arr.size:
            raise ValueError(
                f"Flat image size {arr.size} is not 3*H*W for square image."
            )
        if input_format == 'CHW':
            arr = arr.reshape(3, side, side).transpose(0, 2, 1)
        elif input_format == 'CWH':
            arr = arr.reshape(3, side, side)
        elif input_format == 'HWC':
            arr = arr.reshape(side, side, 3).transpose(2, 1, 0)
        else:
            raise ValueError(f"Unsupported input format: {input_format}")
        return arr

    if arr.ndim == 3:
        if input_format == 'CHW':
            return arr.transpose(0, 2, 1)
        if input_format == 'CWH':
            return arr
        if input_format == 'HWC':
            return arr.transpose(2, 1, 0)
        raise ValueError(f"Unsupported input format: {input_format}")

    raise ValueError(f"Unsupported image rank {arr.ndim}; expected 1-D or 3-D.")


def make_env_from_spec(spec):
    """Create an environment from a spec string.

    spec can be:
      - a gym id string (e.g. StretchMuJoCo-v0) -> will call gym.make(spec)
      - a module:callable string (e.g. mypkg.envs:make_env) -> imports module and calls callable()
    """
    if spec is None:
        raise ValueError('No env spec provided')

    # try module:callable form
    if ':' in spec:
        module_name, factory_name = spec.split(':', 1)
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
        return factory()

    # else try gym make
    try:
        import gym

        return gym.make(spec)
    except Exception as e:
        raise RuntimeError(f"Could not create env from spec '{spec}': {e}")


def get_image_from_obs_or_env(obs, env):
    """Return an image (HWC uint8 or float) either from the obs dict keys or env.render()."""
    # common keys used in different wrappers
    if isinstance(obs, dict):
        for k in ['image_observation', 'image', 'image_obs', 'pixels', 'rgb']:
            if k in obs:
                return obs[k]
    # fallback: try env.render
    if hasattr(env, 'render'):
        try:
            im = env.render(mode='rgb_array')
            return im
        except Exception:
            return None
    return None


def collect(
    env,
    num_episodes,
    horizon,
    policy_factory,
    obs_dict=False,
    render_image=False,
    image_input_format='CHW',
):
    """Collect episodes from env using policy_factory(env, horizon) -> policy_fn(obs).

    Returns a list of path dicts matching the repo expectation.
    """
    paths = []
    for ep in range(num_episodes):
        reset_result = env.reset()
        # Handle both old gym API (returns obs) and new gymnasium API (returns obs, info)
        if isinstance(reset_result, tuple):
            obs, reset_info = reset_result
        else:
            obs = reset_result
            reset_info = {}

        # Create a fresh policy each episode (re-reads start/goal qpos after reset)
        policy_fn = policy_factory(env, horizon=horizon)

        observations = []
        next_observations = []
        actions = []
        rewards = []
        terminals = []
        agent_infos = []
        env_infos = []

        # for t in range(horizon):
        #     # Store observation before taking action
        #     observations.append(obs)
            
        #     action = policy_fn(obs)
        #     step_res = env.step(action)
        #     if len(step_res) == 4:
        #         # Old Gym API: (next_obs, reward, done, info)
        #         next_obs, reward, done, info = step_res
        #         agent_info = {}
        #         env_info = info if isinstance(info, dict) else {}
        #     elif len(step_res) == 5:
        #         # New Gymnasium API: (next_obs, reward, terminated, truncated, info)
        #         next_obs, reward, terminated, truncated, info = step_res
        #         done = terminated or truncated  # Combine for backwards compatibility
        #         agent_info = {}
        #         env_info = info if isinstance(info, dict) else {}
        #     else:
        #         # Broad compatibility: try unpacking first 4
        #         next_obs = step_res[0]
        #         reward = step_res[1]
        #         done = step_res[2]
        #         info = step_res[3] if len(step_res) > 3 else {}
        #         agent_info = {}
        #         env_info = info if isinstance(info, dict) else {}

        #     actions.append(np.array(action))
        #     rewards.append(float(reward))
        #     terminals.append(bool(done))
        #     agent_infos.append(agent_info if isinstance(agent_info, dict) else {})
        #     env_infos.append(env_info if isinstance(env_info, dict) else {})

        #     next_observations.append(next_obs)

        #     obs = next_obs

        #     if done:
        #         break
        for t in range(horizon):
            observations.append(obs)
            action = policy_fn(obs)
            
            step_res = env.step(action)
            # Properly unpack new Gymnasium 5-tuple: (obs, reward, terminated, truncated, info)
            if len(step_res) == 5:
                next_obs, reward, terminated, truncated, info = step_res
                done = terminated or truncated
            elif len(step_res) == 4:
                next_obs, reward, done, info = step_res
                terminated = done
                truncated = False
            else:
                next_obs, reward, done = step_res[0], step_res[1], step_res[2]
                info = step_res[-1] if isinstance(step_res[-1], dict) else {}
                terminated = done
                truncated = False
            info = info if isinstance(info, dict) else {}

            actions.append(np.array(action))
            rewards.append(float(reward))
            terminals.append(bool(done))
            agent_infos.append({})
            env_infos.append(info)
            next_observations.append(next_obs)
            obs = next_obs

            if done:
                break

        # Append the final state so len(observations) == len(actions) + 1.
        # preprocess() in DictToMDPPathLoader does observation[:-1], which
        # drops this sentinel — but H = min(len(obs), len(actions)) - 1
        # requires len(observations) > len(actions).
        observations.append(obs)
        next_observations.append(obs)  # unused by loader but keeps lengths consistent

        # Disable teleport mode after the episode so env.reset() is unaffected
        if hasattr(env, 'unwrapped'):
            env.unwrapped._allow_teleport = False
        elif hasattr(env, '_allow_teleport'):
            env._allow_teleport = False

        # Mark the last step as terminal=True so the loader knows where the episode ends.
        if len(terminals) > 0 and not terminals[-1]:
            terminals[-1] = True

        # If obs_dict=True, the ImageObservationWrapper returns a dict with
        # 'image_observation' and 'image_desired_goal' at every step.
        # DictToMDPPathLoader.preprocess() only reads 'image_observation' and
        # reconstructs all other image keys itself:
        #   - image_desired_goal      ← images[-1]  (last frame = robot at goal)
        #   - image_achieved_goal     ← images[i]   (current frame)
        #   - initial_image_observation ← images[0] (first frame)
        # So we only need to store 'image_observation', flat float32 [0, 1].
        if obs_dict:
            def _to_flat_obs(o):
                """Keep image_observation, converted to CWH then flattened to float32 [0,1]."""
                if not isinstance(o, dict) or 'image_observation' not in o:
                    return o
                img = o['image_observation']
                img = _to_cwh_image(img, input_format=image_input_format).flatten()
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
                return {'image_observation': img}

            observations = [_to_flat_obs(o) for o in observations]
            next_observations = [_to_flat_obs(o) for o in next_observations]

        path = dict(
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            agent_infos=agent_infos,
            env_infos=env_infos,
        )
        paths.append(path)

        print(f"Collected episode {ep+1}/{num_episodes}: length={len(actions)}")

    return paths


def random_policy_factory(action_space):
    def policy(_obs):
        return action_space.sample()

    return policy


def linear_interpolation_policy_factory(env, horizon=75):
    """
    Open-loop linear interpolation policy — directly teleports joints.

    At episode start: read start and goal joint positions from qpos.
    At each step t (0..horizon-1):
        desired = start + (t / (horizon-1)) * (goal - start)
    Writes desired ABSOLUTE positions into env._direct_qpos (teleport branch in step()).

    Action format: normalized absolute position in [-1, 1] per actuator.
        action[i] = clip(2 * (desired[i] - low[i]) / (high[i] - low[i]) - 1, -1, 1)
    This matches step()'s de-normalization:
        ctrl_target = low + (a + 1) / 2 * (high - low)
    so action = normalize_ctrl(desired) is the exact inverse.

    Each step produces a different action_label because desired changes every step.

    _direct_qpos layout: [base_x (m), lift (m), arm_ext (m), gripper (m)]
    """
    unwrapped = env.unwrapped

    start_qpos = unwrapped.data.qpos.copy()
    goal_qpos  = unwrapped._goal_qpos.copy()

    # Actuator physical ranges: [base_pos, lift, arm, gripper]
    low, high = unwrapped.model.actuator_ctrlrange.T  # shape (nu,) each

    def qpos_to_state(q):
        base    = q[0]                       # IDX_BASE_X
        lift    = q[3]                       # IDX_LIFT
        arm     = q[4] + q[5] + q[6] + q[7] # sum of 4 telescope joints
        gripper = q[11]                      # IDX_GRIPPER
        return np.array([base, lift, arm, gripper], dtype=np.float64)

    def normalize_ctrl(ctrl):
        """Map absolute ctrl position → [-1, 1] using actuator ctrlrange."""
        return 2.0 * (ctrl - low) / (high - low) - 1.0

    start = qpos_to_state(start_qpos)
    goal  = qpos_to_state(goal_qpos)

    print(f"[LerpPolicy] ========== EPISODE DEBUG ==========")
    print(f"[LerpPolicy] 1. start_qpos (full, len={len(start_qpos)}):")
    print(f"   {start_qpos}")
    print(f"[LerpPolicy] 2. goal_qpos (full, len={len(goal_qpos)}):")
    print(f"   {goal_qpos}")
    print(f"[LerpPolicy] 3. start [base, lift, arm, gripper]:")
    print(f"   base={start[0]:.6f}, lift={start[1]:.6f}, arm={start[2]:.6f}, gripper={start[3]:.6f}")
    print(f"[LerpPolicy] 4. goal  [base, lift, arm, gripper]:")
    print(f"   base={goal[0]:.6f},  lift={goal[1]:.6f},  arm={goal[2]:.6f},  gripper={goal[3]:.6f}")
    print(f"[LerpPolicy] 5. ctrlrange low : {low}")
    print(f"[LerpPolicy] 6. ctrlrange high: {high}")
    print(f"[LerpPolicy] ======================================")

    # Enable teleport mode so the teleport branch in step() fires.
    unwrapped._allow_teleport = True

    state = {
        'step': 0,
        # Initialize previous target so the first delta/action label is well-defined.
        'prev_desired': start.copy(),
    }

    def policy(obs):
        t = state['step']
        frac = t / (horizon - 1) if horizon > 1 else 1.0
        frac = min(frac, 1.0)

        # Exact desired position at this step
        desired = start + frac * (goal - start)

        # Write absolute joint positions — step() teleports qpos directly
        unwrapped._direct_qpos = desired.astype(np.float32)

        # Compute a meaningful action label: the delta from the previous desired
        # position, normalized by _delta_scale so it matches the RL action space
        # units (action=1 means move one _delta_scale unit).
        # Layout: [base_x (vel proxy), lift, arm_ext, gripper]
        delta = desired - state['prev_desired']
        delta_scale = unwrapped._delta_scale  # shape (4,), same layout

        # Normalize: action[i] = delta[i] / delta_scale[i], clipped to [-1, 1]
        # base (index 0) uses velocity semantics — treat same as others here
        action_label = np.clip(delta / (delta_scale + 1e-8), -1.0, 1.0).astype(np.float32)

        print(f"[LerpPolicy] --- Step {t} (frac={frac:.4f}) ---")
        print(f"  5. desired      : base={desired[0]:.6f}, lift={desired[1]:.6f}, arm={desired[2]:.6f}, gripper={desired[3]:.6f}")
        print(f"  6. delta        : base={delta[0]:.6f}, lift={delta[1]:.6f}, arm={delta[2]:.6f}, gripper={delta[3]:.6f}")
        print(f"  7. delta_scale  : {delta_scale}")
        print(f"  8. action_label : base={action_label[0]:.6f}, lift={action_label[1]:.6f}, arm={action_label[2]:.6f}, gripper={action_label[3]:.6f}")

        state['prev_desired'] = desired.copy()
        state['step'] += 1
        return action_label

    return policy

    
def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--env-id', type=str, help='Gym env id to create via gym.make')
    group.add_argument('--env-factory', type=str,
                       help="Python factory in MODULE:CALLABLE form that returns an env instance")

    parser.add_argument('--num-episodes', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=400)
    parser.add_argument('--save-path', type=str, default='data/stretch_demos.pkl')
    parser.add_argument('--obs-dict', action='store_true', help='Save observations as dicts with image_observation key')
    parser.add_argument('--policy', type=str, default='interpolation', 
                        choices=['random', 'goal_reaching', 'interpolation'],
                        help='Policy to use for collecting data. Default: interpolation')
    parser.add_argument('--policy-gain', type=float, default=0.3,
                        help='Proportional gain for goal-reaching policy (higher = more aggressive)')
    parser.add_argument('--policy-noise', type=float, default=0.1,
                        help='Action noise std for exploration (0.1 = moderate, 0.2 = high)')
    parser.add_argument(
        '--image-input-format',
        type=str,
        choices=['CHW', 'CWH', 'HWC'],
        default='CHW',
        help='Layout of env image_observation before conversion to saved flattened CWH (default: CHW).',
    )
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    spec = args.env_factory if args.env_factory is not None else args.env_id
    env = make_env_from_spec(spec)

    # prepare policy
    if args.policy == 'random':
        # we wrap the old policy in a factory
        policy_fn_or_factory = lambda e, h: random_policy_factory(e.action_space)
    elif args.policy == 'goal_reaching':
        # we wrap the old policy in a factory
        policy_fn_or_factory = lambda e, h: goal_reaching_policy_factory_wrapper(
            e, gain=args.policy_gain, noise_std=args.policy_noise
        )
    elif args.policy == 'interpolation':
        # we wrap the old policy in a factory
        policy_fn_or_factory = linear_interpolation_policy_factory
    else:
        raise ValueError('Unsupported policy: ' + args.policy)

    

    try:
        paths = collect(
            env,
            args.num_episodes,
            args.horizon,
            policy_fn_or_factory,
            obs_dict=args.obs_dict,
            image_input_format=args.image_input_format,
        )

        # ensure save dir exists
        save_dir = os.path.dirname(args.save_path)
        if save_dir != '' and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        with open(args.save_path, 'wb') as f:
            pickle.dump(paths, f)

        print('Saved', len(paths), 'paths to', args.save_path)

        # ---- Debug: print trajectory contents ----
        print("\n========== TRAJECTORY DEBUG ==========")
        for ep_idx, path in enumerate(paths):
            print(f"\n--- Episode {ep_idx} ---")

            def _fmt_obs(v_arr):
                """Print stats + head/mid/tail slices to detect per-step variation."""
                if v_arr.ndim == 1 and v_arr.size > 24:
                    mid = v_arr.size // 2
                    head = np.round(v_arr[:4], 4).tolist()
                    middle = np.round(v_arr[mid:mid+4], 4).tolist()
                    tail = np.round(v_arr[-4:], 4).tolist()
                    chk = hash(v_arr.tobytes()) & 0xFFFFFFFF  # 32-bit fingerprint
                    # Report std and number of unique values — more informative than
                    # fixed head/mid/tail slices which may fall on static background
                    n_unique = len(np.unique(np.round(v_arr, 3)))
                    return (f"shape={v_arr.shape} dtype={v_arr.dtype} "
                            f"min={v_arr.min():.4f} max={v_arr.max():.4f} mean={v_arr.mean():.4f} "
                            f"hash=0x{chk:08x} | "
                            f"head={head} mid[{mid}:{mid+4}]={middle} tail={tail}")
                return f"shape={v_arr.shape} dtype={v_arr.dtype} values={v_arr}"

            # observations
            print(f"  observations ({len(path['observations'])} steps):")
            for i, o in enumerate(path['observations']):
                if isinstance(o, dict):
                    print(f"    [step {i}]:")
                    for k, v in o.items():
                        print(f"      '{k}': {_fmt_obs(np.asarray(v))}")
                else:
                    print(f"    [step {i}]: {_fmt_obs(np.asarray(o))}")

            # next_observations
            print(f"  next_observations ({len(path['next_observations'])} steps):")
            for i, o in enumerate(path['next_observations']):
                if isinstance(o, dict):
                    print(f"    [step {i}]:")
                    for k, v in o.items():
                        print(f"      '{k}': {_fmt_obs(np.asarray(v))}")
                else:
                    print(f"    [step {i}]: {_fmt_obs(np.asarray(o))}")

            # actions
            print(f"  actions ({len(path['actions'])} steps):")
            for i, a in enumerate(path['actions']):
                print(f"    [step {i}]: {np.asarray(a)}")

            # rewards
            print(f"  rewards ({len(path['rewards'])} steps):")
            for i, r in enumerate(path['rewards']):
                print(f"    [step {i}]: {r}")

            # terminals
            print(f"  terminals ({len(path['terminals'])} steps):")
            for i, t in enumerate(path['terminals']):
                print(f"    [step {i}]: {t}")

            # agent_infos
            print(f"  agent_infos ({len(path['agent_infos'])} steps):")
            for i, ai in enumerate(path['agent_infos']):
                print(f"    [step {i}]: {ai}")

            # env_infos
            print(f"  env_infos ({len(path['env_infos'])} steps):")
            for i, ei in enumerate(path['env_infos']):
                print(f"    [step {i}]: {ei}")

            print("=======================================\n")
    finally:
        # Explicit close avoids relying on __del__ cleanup at interpreter shutdown.
        if hasattr(env, 'close'):
            try:
                env.close()
            except Exception as e:
                print(f"[WARN] env.close() failed: {e}")


if __name__ == '__main__':
    main()
