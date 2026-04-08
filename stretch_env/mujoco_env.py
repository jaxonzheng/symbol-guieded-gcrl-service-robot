from os import path
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import error, logger, spaces
from gymnasium.spaces import Space
from numpy.typing import NDArray

try:
    import mujoco
except ImportError as e:
    MUJOCO_IMPORT_ERROR = e
else:
    MUJOCO_IMPORT_ERROR = None


DEFAULT_SIZE = 480 # render size 480*480


def expand_model_path(model_path: str) -> str:
    """Expands the `model_path` to a full path if it starts with '~' or '.' or '/'."""
    if model_path.startswith(".") or model_path.startswith("/"):
        fullpath = model_path
    elif model_path.startswith("~"):
        fullpath = path.expanduser(model_path)
    else:
        fullpath = path.join(path.dirname(__file__), "assets", model_path)
    if not path.exists(fullpath):
        raise OSError(f"File {fullpath} does not exist")

    return fullpath


class BaseMujocoEnv(gym.Env[NDArray[np.float64], NDArray[np.float32]]):
    """Superclass for all MuJoCo environments."""

    def __init__(
        self,
        model_path,
        frame_skip,
        observation_space: Optional[Space],
        render_mode: Optional[str] = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ):
        self.fullpath = expand_model_path(model_path)

        self.width = width
        self.height = height
        self.model, self.data = self._initialize_simulation()

        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        self.frame_skip = frame_skip

        assert self.metadata["render_modes"] == [
            "human",
            "rgb_array",
            "depth_array",
        ], self.metadata["render_modes"]
        if "render_fps" in self.metadata:
            assert (
                int(np.round(1.0 / self.dt)) == self.metadata["render_fps"]
            ), f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata["render_fps"]}'
        if observation_space is not None:
            self.observation_space = observation_space
        self._set_action_space()

        self.render_mode = render_mode
        self.camera_name = camera_name
        self.camera_id = camera_id

    def _set_action_space(self):
        # bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        # low, high = bounds.T
        # self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # self.action_space = spaces.Box(
        # low=-1.0,
        # high=1.0,
        # shape=(self.model.nu,),
        # dtype=np.float32
        # )
        nu = self.model.nu  # number of actuators
        low  = -np.ones(nu, dtype=np.float32)  # all -1
        high =  np.ones(nu, dtype=np.float32)  # all +1
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def step(
        self, action: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float64], np.float64, bool, bool, Dict[str, np.float64]]:
        raise NotImplementedError

    def reset_model(self) -> NDArray[np.float64]:
        raise NotImplementedError

    def _initialize_simulation(self) -> Tuple[Any, Any]:
        raise NotImplementedError

    def _reset_simulation(self) -> None:
        raise NotImplementedError

    def _step_mujoco_simulation(self, ctrl, n_frames) -> None:
        raise NotImplementedError

    def render(self) -> Union[NDArray[np.float64], None]:
        raise NotImplementedError

    def _get_reset_info(self) -> Dict[str, float]:
        return {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._reset_simulation()
        ob = self.reset_model()
        info = self._get_reset_info()
        if self.render_mode == "human":
            self.render()
        return ob, info

    def set_state(self, qpos, qvel) -> None:
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)

    @property
    def dt(self) -> float:
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames) -> None:
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(
                f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}"
            )
        self._step_mujoco_simulation(ctrl, n_frames)

    def close(self):
        raise NotImplementedError

    def get_body_com(self, body_name) -> NDArray[np.float64]:
        raise NotImplementedError

    def state_vector(self) -> NDArray[np.float64]:
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])


class MujocoEnv(BaseMujocoEnv):
    def __init__(
        self,
        model_path: str,
        frame_skip: int,
        observation_space: Space,
        render_mode: Optional[str] = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
        default_camera_config: Optional[Dict[str, Union[float, int]]] = None, # for camera configuration
    ):
        if MUJOCO_IMPORT_ERROR is not None:
            raise error.DependencyNotInstalled(
                f"{MUJOCO_IMPORT_ERROR}. "
                "Install mujoco with `pip install mujoco`"
            )

        self.mujoco_renderer = None
        self.default_camera_config = default_camera_config # for camera configuration

        super().__init__(
            model_path,
            frame_skip,
            observation_space,
            render_mode,
            width,
            height,
            camera_id,
            camera_name,
        )

    def _initialize_simulation(self):
        # MuJoCo 3.1.6 supports all XML features directly
        model = mujoco.MjModel.from_xml_path(self.fullpath)
        data = mujoco.MjData(model)
        return model, data

    def _reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)

    def set_state(self, qpos, qvel):
        super().set_state(qpos, qvel)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def get_body_com(self, body_name):
        return self.data.body(body_name).xpos

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl

        mujoco.mj_step(self.model, self.data, nstep=n_frames)

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.mujoco_renderer is None:
            from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
            
            # Determine camera for rendering
            camera_id = self.camera_id
            camera_name = self.camera_name

            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None and camera_name is not None:
                camera_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    camera_name,
                )
            
            # Pass camera configuration to the renderer during initialization
            self.mujoco_renderer = MujocoRenderer(
                self.model, 
                self.data, 
                self.default_camera_config,
                width=640,
                height=480,
                camera_id=camera_id,
                camera_name=camera_name if camera_id is None else None,
            )

        if self.render_mode == "rgb_array":
            return self.mujoco_renderer.render(self.render_mode)
        elif self.render_mode == "depth_array":
            return self.mujoco_renderer.render(self.render_mode)
            # self.mujoco_renderer.render(
            #     self.render_mode, camera_id=camera_id, camera_name=camera_name
            # )
        elif self.render_mode == "human":
            self.mujoco_renderer.render(self.render_mode)

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position and so forth.
        """
        pass