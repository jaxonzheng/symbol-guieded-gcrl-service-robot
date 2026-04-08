"""
Image Observation Wrapper for Stretch Pick Environment

This wrapper converts state-based observations to image-based observations
required by Stable Contrastive RL training code.

Original format: Box(shape=(290,), dtype=float32) - concatenated [state, goal]
Wrapped format: Dict with keys:
    - 'image_observation': (3, 48, 48) uint8 - current state rendered as RGB image
    - 'image_desired_goal': (3, 48, 48) uint8 - goal state rendered as RGB image
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ImageObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper that converts state vector observations to image-based observations.
    
    The base environment returns concatenated [current_state, goal_state] as a flat vector.
    This wrapper:
    1. Renders the current state as an RGB image
    2. Renders the goal state as an RGB image (by temporarily setting qpos to goal_qpos)
    3. Returns a dictionary with both images in CWH format (channels first)
    
    Args:
        env: Base StretchPickEnv environment
        image_width: Width of rendered images (default: 48)
        image_height: Height of rendered images (default: 48)
        camera_name: Name of camera to render from (default: 'head_camera')
    """
    
    def __init__(self, env, image_width=48, image_height=48, camera_name='head_camera'):
        super().__init__(env)
        
        self.image_width = image_width
        self.image_height = image_height
        self.camera_name = camera_name
        
        # Set camera on base environment BEFORE renderer initialization
        # The renderer picks up the camera during first render() call
        self.env.camera_name = camera_name
        self.env.camera_id = None  # Use name, not ID
        
        # Close any existing renderer to force reinitialization with new camera
        if hasattr(self.env, 'mujoco_renderer') and self.env.mujoco_renderer is not None:
            self.env.mujoco_renderer.close()
            self.env.mujoco_renderer = None
        
        # Define new observation space - dict with two image keys
        self.observation_space = spaces.Dict({
            'image_observation': spaces.Box(
                low=0, high=255,
                shape=(3, image_height, image_width),  # CWH format (channels first)
                dtype=np.uint8
            ),
            'image_desired_goal': spaces.Box(
                low=0, high=255,
                shape=(3, image_height, image_width),  # CWH format (channels first)
                dtype=np.uint8
            ),
        })
        
        print(f"[ImageObservationWrapper] Initialized with image size {image_width}x{image_height}, camera '{camera_name}'")
        print(f"[ImageObservationWrapper] New observation space: {self.observation_space}")
    
    def observation(self, obs):
        """
        Convert state vector observation to image-based observation dictionary.
        
        Args:
            obs: Unused - we render directly from environment state
        
        Returns:
            dict with keys 'image_observation' and 'image_desired_goal'
        """
        # Render current state
        current_image = self._render_current_state()
        
        # Render goal state
        goal_image = self._render_goal_state()
        
        return {
            'image_observation': current_image,
            'image_desired_goal': goal_image
        }
    
    def _render_current_state(self):
        """
        Render the current state as an RGB image.
        
        Returns:
            np.ndarray: RGB image in CWH format (3, H, W), dtype uint8
        """
        # Initialize renderer if needed by calling render() once
        # The renderer will be initialized with the camera set in __init__
        if self.env.mujoco_renderer is None:
            _ = self.env.render()  # This initializes the renderer with the camera
        
        # Save current mocap position to restore after rendering
        import mujoco
        mocap_id = int(self.env.model.body_mocapid[
            mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "target0")
        ])
        mocap_pos_saved = self.env.data.mocap_pos[mocap_id].copy()
        
        # Update mocap position to be above current object position
        # This ensures the red dot visualization is always correct
        # Get current object position from qpos (IDX_OBJ_X = 21 for object0)
        obj_x_current = self.env.data.qpos[21]  # IDX_OBJ_X
        obj_y_current = self.env.data.qpos[22]  # IDX_OBJ_Y
        obj_z_current = self.env.data.qpos[23]  # IDX_OBJ_Z
        
        # Position mocap at target position (slightly in front of object)
        APPROACH_OFFSET_Y = -0.025  # Same offset used in IK computation
        target_x = obj_x_current
        target_y = obj_y_current + APPROACH_OFFSET_Y
        target_z = obj_z_current
        
        self.env.data.mocap_pos[mocap_id] = np.array([target_x, target_y, target_z])
        
        # Update simulation state to reflect mocap position change
        # Use the same complete update chain that works for goal state
        mujoco.mj_kinematics(self.env.model, self.env.data)
        mujoco.mj_comPos(self.env.model, self.env.data)
        mujoco.mj_camlight(self.env.model, self.env.data)
        mujoco.mj_tendon(self.env.model, self.env.data)
        mujoco.mj_transmission(self.env.model, self.env.data)
        
        # Render from current state
        image = self.env.mujoco_renderer.render('rgb_array')

        # DEBUG: print raw render output
        print(f"[_render_current_state] RAW HWC {image.shape} {image.dtype} "
              f"min={image.min()} max={image.max()} mean={image.mean():.2f} | "
              f"top-left pixel (RGB)={image[0, 0, :].tolist()} | "
              f"center pixel (RGB)={image[image.shape[0]//2, image.shape[1]//2, :].tolist()}")

        # Restore original mocap position
        self.env.data.mocap_pos[mocap_id] = mocap_pos_saved
        
        # Restore simulation state
        mujoco.mj_kinematics(self.env.model, self.env.data)
        mujoco.mj_comPos(self.env.model, self.env.data)
        mujoco.mj_camlight(self.env.model, self.env.data)
        mujoco.mj_tendon(self.env.model, self.env.data)
        mujoco.mj_transmission(self.env.model, self.env.data)
        
        # Resize if needed
        if image.shape[:2] != (self.image_height, self.image_width):
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image)
            pil_img = pil_img.resize((self.image_width, self.image_height), PILImage.BILINEAR)
            image = np.array(pil_img)
        
        # image is (H, W, 3) in HWC format
        # Convert to CWH format (channels first) as required by training code
        image_cwh = np.transpose(image, (2, 0, 1))  # (H, W, 3) -> (3, H, W)
        
        # Ensure uint8 dtype
        if image_cwh.dtype != np.uint8:
            image_cwh = image_cwh.astype(np.uint8)

        # DEBUG: print final CWH output
        # Also show flattened float32 [0,1] vector (first 12 values) — exactly what gets stored in pkl
        flat_f32 = image_cwh.flatten().astype(np.float32) / 255.0
        print(f"[_render_current_state] CWH {image_cwh.shape} {image_cwh.dtype} "
              f"min={image_cwh.min()} max={image_cwh.max()} mean={image_cwh.mean():.2f} | "
              f"flat shape=({flat_f32.shape[0]},) expected=(6912,) match={flat_f32.shape[0]==6912} | "
              f"first 12 float32 vals={np.round(flat_f32[:12], 4).tolist()}")

        return image_cwh
    
    def _render_goal_state(self):
        """
        Render the goal state as an RGB image.
        
        This temporarily sets the environment to the goal state, renders it,
        then restores the original state.
        
        Returns:
            np.ndarray: RGB image in CWH format (3, H, W), dtype uint8
        """
        # Check if goal qpos is stored
        if not hasattr(self.env, '_goal_qpos') or self.env._goal_qpos is None:
            raise RuntimeError(
                "Goal qpos not found in environment. "
                "Make sure the environment stores _goal_qpos during goal generation."
            )
        
        # Initialize renderer if needed
        if self.env.mujoco_renderer is None:
            _ = self.env.render()  # This initializes the renderer
        
        # Save current state (including mocap)
        qpos_current = self.env.data.qpos.copy()
        qvel_current = self.env.data.qvel.copy()
        mocap_pos_current = self.env.data.mocap_pos.copy()
        
        # Set to goal state
        self.env.data.qpos[:] = self.env._goal_qpos
        self.env.data.qvel[:] = self.env._goal_qvel if hasattr(self.env, '_goal_qvel') else 0.0
        
        # Set mocap position based on object position in goal state
        # This ensures red dot is always above the object
        import mujoco
        mocap_id = int(self.env.model.body_mocapid[
            mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "target0")
        ])
        
        # Get object position from goal qpos (IDX_OBJ_X = 21 for object0)
        # Object position is stored at indices 21, 22, 23 (x, y, z)
        obj_x_goal = self.env._goal_qpos[21]  # IDX_OBJ_X
        obj_y_goal = self.env._goal_qpos[22]  # IDX_OBJ_Y
        obj_z_goal = self.env._goal_qpos[23]  # IDX_OBJ_Z
        
        # Position mocap at target position (slightly in front of object, matching IK computation)
        APPROACH_OFFSET_Y = -0.025  # Same offset used in IK computation
        target_x = obj_x_goal
        target_y = obj_y_goal + APPROACH_OFFSET_Y
        target_z = obj_z_goal
        
        self.env.data.mocap_pos[mocap_id] = np.array([target_x, target_y, target_z])
        
        # Need to properly update the simulation state for rendering
        # mj_kinematics: compute positions based on qpos
        # mj_comPos: compute center of mass positions
        # mj_camlight: update camera and light positions
        # mj_tendon: update tendon lengths
        # mj_transmission: compute actuator transmission
        import mujoco
        mujoco.mj_kinematics(self.env.model, self.env.data)
        mujoco.mj_comPos(self.env.model, self.env.data)
        mujoco.mj_camlight(self.env.model, self.env.data)
        mujoco.mj_tendon(self.env.model, self.env.data)
        mujoco.mj_transmission(self.env.model, self.env.data)
        
        # Render goal state
        goal_image = self.env.mujoco_renderer.render('rgb_array')
        
        # Resize if needed
        if goal_image.shape[:2] != (self.image_height, self.image_width):
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(goal_image)
            pil_img = pil_img.resize((self.image_width, self.image_height), PILImage.BILINEAR)
            goal_image = np.array(pil_img)
        
        # Restore original state (including mocap)
        self.env.data.qpos[:] = qpos_current
        self.env.data.qvel[:] = qvel_current
        self.env.data.mocap_pos[:] = mocap_pos_current
        
        # Properly restore all simulation state components
        mujoco.mj_kinematics(self.env.model, self.env.data)
        mujoco.mj_comPos(self.env.model, self.env.data)
        mujoco.mj_camlight(self.env.model, self.env.data)
        mujoco.mj_tendon(self.env.model, self.env.data)
        mujoco.mj_transmission(self.env.model, self.env.data)
        
        # Convert to CWH format
        goal_image_cwh = np.transpose(goal_image, (2, 0, 1))  # (H, W, 3) -> (3, H, W)
        
        # Ensure uint8 dtype
        if goal_image_cwh.dtype != np.uint8:
            goal_image_cwh = goal_image_cwh.astype(np.uint8)
        
        return goal_image_cwh
    
    def reset(self, **kwargs):
        """Reset environment and return image-based observation."""
        obs, info = self.env.reset(**kwargs)
        
        # Convert to image observation
        obs_dict = self.observation(obs)
        
        print(f"[ImageObservationWrapper] Reset - image_observation shape: {obs_dict['image_observation'].shape}")
        print(f"[ImageObservationWrapper] Reset - image_desired_goal shape: {obs_dict['image_desired_goal'].shape}")
        
        return obs_dict, info
    
    def step(self, action):
        """Step environment and return image-based observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Convert to image observation
        obs_dict = self.observation(obs)
        
        return obs_dict, reward, terminated, truncated, info


def make_image_observation_env(env, **wrapper_kwargs):
    """
    Convenience function to wrap an environment with ImageObservationWrapper.
    
    Args:
        env: Base environment (StretchPickEnv)
        **wrapper_kwargs: Additional arguments for ImageObservationWrapper
            - image_width: Width of rendered images (default: 48)
            - image_height: Height of rendered images (default: 48)
            - camera_name: Name of camera to render from (default: 'head_camera')
    
    Returns:
        Wrapped environment with image observations
    
    Example:
        >>> from stretch_env.stretch_pick_env import StretchPickEnv
        >>> from stretch_env.image_observation_wrapper import make_image_observation_env
        >>> 
        >>> base_env = StretchPickEnv(
        ...     xml_path='path/to/scene.xml',
        ...     num_objects=5,
        ...     render_mode='rgb_array'
        ... )
        >>> env = make_image_observation_env(base_env, image_width=48, image_height=48)
    """
    return ImageObservationWrapper(env, **wrapper_kwargs)
