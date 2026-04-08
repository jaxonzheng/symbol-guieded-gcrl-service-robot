#!/usr/bin/env python3
"""
Factory to create the Stretch MuJoCo environment for stable_contrastive_rl.

This uses the local copy of the Stretch environment in stretch_env/
(copied from sgcrl to keep the environments separate).

Use with the stable_contrastive_rl conda environment:
    conda activate stable_contrastive_rl
"""
import sys
import os

# Add parent directory to path so we can import stretch_env
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def make_stretch_pick_env(num_objects=1, render_mode='rgb_array', camera_name='table_view'):
    """Create a StretchPickEnv with image observations.
    
    Args:
        num_objects: number of objects in the scene (default: 1)
        render_mode: 'rgb_array' or 'human'
        camera_name: which camera to use for rendering
    
    Returns:
        StretchPickEnv instance
    """
    from stretch_env import StretchPickEnv
    
    env = StretchPickEnv(
        num_objects=num_objects,
        render_mode=render_mode,
    )
    
    # Store camera name for rendering
    env._camera_name = camera_name
    
    return env


def make_stretch_pick_env_with_image_obs(num_objects=1, camera_name='table_view', image_size=(64, 64)):
    """Create a StretchPickEnv wrapped to return dict observations with images.
    
    Returns dict observations with:
        - 'state_observation': the original state vector
        - 'image_observation': RGB image from the camera
    """
    from stretch_env import StretchPickEnv
    from gymnasium import ObservationWrapper
    from gymnasium.spaces import Dict, Box
    import numpy as np
    
    class ImageObsWrapper(ObservationWrapper):
        def __init__(self, env, camera_name='table_view', image_size=(64, 64)):
            super().__init__(env)
            self.camera_name = camera_name
            self.image_size = image_size
            
            # Update observation space to dict
            self.observation_space = Dict({
                'state_observation': env.observation_space,
                'image_observation': Box(
                    low=0, high=255, 
                    shape=(image_size[0], image_size[1], 3), 
                    dtype=np.uint8
                ),
            })
        
        def observation(self, obs):
            # Render image
            img = self.env.mujoco_renderer.render(
                render_mode='rgb_array',
                camera_name=self.camera_name
            )
            
            # Resize if needed
            if img.shape[:2] != self.image_size:
                from PIL import Image
                img_pil = Image.fromarray(img)
                img_pil = img_pil.resize((self.image_size[1], self.image_size[0]))
                img = np.array(img_pil)
            
            return {
                'state_observation': obs,
                'image_observation': img,
            }
    
    base_env = StretchPickEnv(
        num_objects=num_objects,
        render_mode='rgb_array',
    )
    
    return ImageObsWrapper(base_env, camera_name=camera_name, image_size=image_size)


def make_stretch_pick_env_with_images(
    num_objects=1, 
    image_width=48, 
    image_height=48, 
    # camera_name='table_view',
    camera_name='left_front_view',
    # camera_name='head_front_down_view'
):
    """
    Create a StretchPickEnv with full image-based observations for Stable Contrastive RL.
    
    Returns dict observations with:
        - 'image_observation': (3, H, W) uint8 - current state as RGB image (CWH format)
        - 'image_desired_goal': (3, H, W) uint8 - goal state as RGB image (CWH format)
    
    This matches the format required by train_eval_stable_contrastive_rl.py:
        - obs_dict=True
        - use_image=True
        - imsize=48 (default)
        - output_image_format='CWH' (channels first)
    
    Args:
        num_objects: Number of objects in the scene (default: 1)
        image_width: Width of rendered images (default: 48)
        image_height: Height of rendered images (default: 48)
        camera_name: Name of camera to render from (default: 'table_view')
    
    Returns:
        StretchPickEnv wrapped with ImageObservationWrapper
    
    Example:
        >>> from scripts.stretch_env_factory import make_stretch_pick_env_with_images
        >>> env = make_stretch_pick_env_with_images(num_objects=5, image_width=48, image_height=48)
        >>> obs, info = env.reset()
        >>> print(obs.keys())  # dict_keys(['image_observation', 'image_desired_goal'])
        >>> print(obs['image_observation'].shape)  # (3, 48, 48)
    """
    from stretch_env import StretchPickEnv
    from stretch_env.image_observation_wrapper import ImageObservationWrapper
    
    # Create base environment with rgb_array rendering
    base_env = StretchPickEnv(
        num_objects=num_objects,
        render_mode='rgb_array',
    )
    
    # Wrap with image observation wrapper
    env = ImageObservationWrapper(
        base_env,
        image_width=image_width,
        image_height=image_height,
        camera_name=camera_name
    )
    
    print(f"[Factory] Created StretchPickEnv with {num_objects} objects")
    print(f"[Factory] Image observations: {image_width}x{image_height} from camera '{camera_name}'")
    print(f"[Factory] Observation space: {env.observation_space}")
    
    return env


if __name__ == '__main__':
    # Test the factory
    print("=" * 60)
    print("Testing make_stretch_pick_env_with_images (for Stable Contrastive RL)")
    print("=" * 60)
    
    env = make_stretch_pick_env_with_images(num_objects=5, image_width=48, image_height=48)
    obs, info = env.reset()
    
    print("\nEnvironment created successfully!")
    print(f"Observation keys: {list(obs.keys())}")
    print(f"image_observation shape: {obs['image_observation'].shape}, dtype: {obs['image_observation'].dtype}")
    print(f"image_desired_goal shape: {obs['image_desired_goal'].shape}, dtype: {obs['image_desired_goal'].dtype}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test a step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"\nStep executed successfully!")
    print(f"Reward: {reward}, Success: {info.get('success', False)}")
    
    print("\n" + "=" * 60)
    print("Old test: make_stretch_pick_env_with_image_obs")
    print("=" * 60)
    env_old = make_stretch_pick_env_with_image_obs(num_objects=1)
    obs_old, info_old = env_old.reset()
    print("Environment created successfully!")
    print(f"Observation keys: {list(obs_old.keys())}")
    print(f"State shape: {obs_old['state_observation'].shape}")
    print(f"Image shape: {obs_old['image_observation'].shape}")
    print(f"Action space: {env_old.action_space}")
