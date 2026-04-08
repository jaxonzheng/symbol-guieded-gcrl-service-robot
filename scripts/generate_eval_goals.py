#!/usr/bin/env python3
"""
Generate eval_goals.pkl for Stable Contrastive RL with StretchPickEnv.

PresampledPathDistribution expects a dict:
  {
    'image_desired_goal': np.array of shape (N, flat_image_size),  # CWH flattened
    'object_xyz': np.array of shape (N, 3),                        # object position in world frame
  }

where flat_image_size = C * W * H = 3 * 48 * 48 = 6912 (float32, values in [0, 1]).

Usage:
  conda run -p /work/rleap1/jaxon.cheng/venvs/stable_contrastive_rl \
    python scripts/generate_eval_goals.py \
    --num-goals 500 \
    --save-path data/stretch_align/eval_goals.pkl \
    --goal-image-input-format CHW \
    --camera-name left_front_view
"""
import argparse
import importlib
import inspect
import os
import pickle
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _to_cwh_image(img, input_format, image_size):
    """
    Convert image to CWH layout (channels, width, height).

    Supported input formats:
      - CHW: (C, H, W) or flattened CHW
      - CWH: (C, W, H) or flattened CWH
      - HWC: (H, W, C) or flattened HWC
    """
    arr = np.asarray(img)

    if arr.ndim == 1:
        if arr.size != 3 * image_size * image_size:
            raise ValueError(
                f"Flat image size mismatch: got {arr.size}, expected {3 * image_size * image_size}"
            )
        if input_format == 'CHW':
            arr = arr.reshape(3, image_size, image_size).transpose(0, 2, 1)
        elif input_format == 'CWH':
            arr = arr.reshape(3, image_size, image_size)
        elif input_format == 'HWC':
            arr = arr.reshape(image_size, image_size, 3).transpose(2, 1, 0)
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

    raise ValueError(f"Unsupported image rank {arr.ndim}; expected 1-D or 3-D image array.")


def _configure_mujoco_gl(mujoco_gl_arg: str) -> str:
    """
    Configure MUJOCO_GL before importing MuJoCo/Gym env modules.
    Priority:
      1) Existing MUJOCO_GL env var
      2) --mujoco-gl CLI arg (if not auto)
      3) Auto-select ('glfw' with display, else 'egl')
    """
    existing = os.environ.get('MUJOCO_GL')
    if existing:
        print(f'[RenderConfig] Using existing MUJOCO_GL={existing}')
        return existing

    if mujoco_gl_arg != 'auto':
        selected = mujoco_gl_arg
    else:
        has_display = bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
        selected = 'glfw' if has_display else 'egl'

    os.environ['MUJOCO_GL'] = selected

    if selected in ('egl', 'osmesa') and 'PYOPENGL_PLATFORM' not in os.environ:
        os.environ['PYOPENGL_PLATFORM'] = selected

    print(f'[RenderConfig] Set MUJOCO_GL={selected}')
    if 'PYOPENGL_PLATFORM' in os.environ:
        print(f"[RenderConfig] PYOPENGL_PLATFORM={os.environ['PYOPENGL_PLATFORM']}")
    return selected


def main():
    parser = argparse.ArgumentParser(description='Generate presampled eval goals for SCRL')
    parser.add_argument('--num-goals', type=int, default=500,
                        help='Number of goal images to sample (default: 500)')
    parser.add_argument('--save-path', type=str, default='data/stretch_goals/eval_goals.pkl',
                        help='Where to save the goals pkl')
    parser.add_argument('--image-size', type=int, default=48,
                        help='Image width/height in pixels (default: 48)')
    parser.add_argument(
        '--goal-image-input-format',
        type=str,
        choices=['CHW', 'CWH', 'HWC'],
        default='CHW',
        help='Layout of env-provided goal images before conversion to saved CWH format (default: CHW).',
    )
    parser.add_argument('--camera-name', type=str, default='left_front_view',
                        help='Camera to render from')
    parser.add_argument('--env-factory', type=str,
                        default='scripts.stretch_env_factory:make_stretch_pick_env_with_images',
                        help='module:factory to instantiate the env')
    parser.add_argument(
        '--mujoco-gl',
        type=str,
        choices=['auto', 'egl', 'osmesa', 'glfw'],
        default='auto',
        help='MuJoCo OpenGL backend. Default auto picks glfw with display, else egl.',
    )
    args = parser.parse_args()
    _configure_mujoco_gl(args.mujoco_gl)

    # Create env
    print(f'Creating environment via {args.env_factory} ...')
    module_name, factory_name = args.env_factory.split(':', 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    # Pass camera and image size into factory when supported.
    factory_sig = inspect.signature(factory)
    factory_kwargs = {}
    if 'camera_name' in factory_sig.parameters:
        factory_kwargs['camera_name'] = args.camera_name
    if 'image_width' in factory_sig.parameters:
        factory_kwargs['image_width'] = args.image_size
    if 'image_height' in factory_sig.parameters:
        factory_kwargs['image_height'] = args.image_size
    env = factory(**factory_kwargs)
    print(f'Factory kwargs: {factory_kwargs}')

    flat_size = 3 * args.image_size * args.image_size  # CWH
    goal_images = np.zeros((args.num_goals, flat_size), dtype=np.float32)
    goal_object_xyz = np.zeros((args.num_goals, 3), dtype=np.float32)
    base_env = getattr(env, 'unwrapped', env)
    target_obj_id = int(getattr(base_env, 'target_obj_id', 0))

    print(f'Sampling {args.num_goals} goal images...')
    for i in range(args.num_goals):
        obs, info = env.reset()
        img_goal_raw = obs['image_desired_goal']
        img_goal_cwh = _to_cwh_image(
            img_goal_raw,
            input_format=args.goal_image_input_format,
            image_size=args.image_size,
        )
        img_flat = img_goal_cwh.flatten()
        if img_flat.dtype == np.uint8:
            img_flat = img_flat.astype(np.float32) / 255.0
        goal_images[i] = img_flat  # flat CWH float32 [0, 1]

        obj_qpos_start = 21 + target_obj_id * 7
        obj_xyz = np.asarray(
            base_env.data.qpos[obj_qpos_start:obj_qpos_start + 3],
            dtype=np.float32,
        )
        goal_object_xyz[i] = obj_xyz

        if (i + 1) % 50 == 0:
            print(f'  sampled {i+1}/{args.num_goals}')

    goals_dict = {
        'image_desired_goal': goal_images,  # (N, 6912) float32 [0, 1]
        'object_xyz': goal_object_xyz,      # (N, 3) float32 world coordinates
    }

    # Save
    save_path = args.save_path
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(goals_dict, f)

    print(f'\nSaved {args.num_goals} goals to: {save_path}')
    print(f'  image_desired_goal shape: {goal_images.shape}')
    print(f'  object_xyz shape: {goal_object_xyz.shape}')
    print(f'  dtype: {goal_images.dtype}')


if __name__ == '__main__':
    main()
