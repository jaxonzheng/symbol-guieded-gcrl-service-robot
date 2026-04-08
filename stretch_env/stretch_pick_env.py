import numpy as np
from gymnasium import utils
from gymnasium.spaces import Box
from .mujoco_env import MujocoEnv
import mujoco
import copy
from .ik.mink_ik import MinkIK



#----------------------
# Base joint: qpos[0] = [x] (only x-axis slide joint)
IDX_BASE_X      = 0  # base_x (slide joint) - ONLY degree of freedom for base

# Robot joints (shifted down by 6 from freejoint version)
IDX_RIGHT_WHEEL = 1
IDX_LEFT_WHEEL  = 2
IDX_LIFT        = 3                 # joint_lift
IDX_ARM_L3      = 4                 # joint_arm_l3 (telescope segment)
IDX_ARM_L2      = 5                 # joint_arm_l2 (telescope segment)
IDX_ARM_L1      = 6                 # joint_arm_l1 (telescope segment)
IDX_ARM_L0      = 7                 # joint_arm_l0 (telescope segment)
IDX_WRIST_YAW   = 8                 # joint_wrist_yaw
IDX_WRIST_PITCH = 9                 # joint_wrist_pitch
IDX_WRIST_ROLL  = 10                # joint_wrist_roll
IDX_GRIPPER     = 11                # joint_gripper_slide
IDX_GRIPPER_LEFT = 12               # joint_gripper_finger_left_open
IDX_RUBBER_LEFT_X = 13
IDX_RUBBER_LEFT_Y = 14
IDX_GRIPPER_RIGHT = 15              # joint_gripper_finger_right_open
IDX_RUBBER_RIGHT_X = 16
IDX_RUBBER_RIGHT_Y = 17
IDX_HEAD_PAN    = 18
IDX_HEAD_TILT   = 19
IDX_HEAD_NAV    = 20

# Object freejoint: qpos[21:28] (shifted down by 6 from original 27)
IDX_OBJ_X       = 21                # object0:joint x
IDX_OBJ_Y       = 22                # object0:joint y
IDX_OBJ_Z       = 23                # object0:joint z
IDX_OBJ_QW      = 24
IDX_OBJ_QX      = 25
IDX_OBJ_QY      = 26
IDX_OBJ_QZ      = 27  
#----------------------

class StretchPickEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 10,
    }

    MAX_OBJECTS = 20
    OBJ_FEATS = 7   # x, y, z, qw, qx, qy, qz
    ROBOT_FEATS = 5  # base_x, lift, arm_ext, gripper, is_grasped (removed wrist_yaw, wrist_pitch, wrist_roll)

    def __init__(self, num_objects=None, objects=None, fixed_start_end=None, **kwargs):

        if objects is not None:
        #ex:[10, 15]
            self.objects = list(objects)
            self.num_objects = len(self.objects)
        else:
        # if no number, then use 1
            if num_objects is None:
                num_objects = 1
            self.num_objects = num_objects
            self.objects = list(range(num_objects))
        
        print(f"[StretchPickEnv] Initializing with num_objects={self.num_objects}, objects={self.objects}")
        
        assert 1 <= self.num_objects <= self.MAX_OBJECTS, \
            f"num_objects must be 1-{self.MAX_OBJECTS}"
        assert max(self.objects) < self.MAX_OBJECTS, \
            f"object id {max(self.objects)} exceeds MAX_OBJECTS={self.MAX_OBJECTS}"
       
        self._fixed_start_end = fixed_start_end
        self._goal = None  # Will be set on first reset
        self._next_reset_object_xyz = None
        

        utils.EzPickle.__init__(self, num_objects=self.num_objects, **kwargs)
        
        # Remove camera parameters from kwargs to avoid conflicts
        # The parent class will set camera_name='track' by default
        kwargs_filtered = {k: v for k, v in kwargs.items() if k not in ['camera_id', 'camera_name']}

        # Define normalization bounds for observations
        # Robot: base_x, lift, arm_ext, gripper, is_grasped
        self._obs_robot_low = np.array([-1.5, 0.53, 0.0, -0.02, 0.0], dtype=np.float32)
        self._obs_robot_high = np.array([1.5, 1.1, 0.52, 0.04, 1.0], dtype=np.float32)
        
        # Objects: x, y, z, qw, qx, qy, qz (per object)
        # Table bounds: x=[-0.5, 0.5], y=[-0.95, -0.45], z=[0.72, 1.0]
        # Quaternions already in [-1, 1]
        self._obs_obj_low = np.array([-0.6, -1.0, 0.7, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        self._obs_obj_high = np.array([0.6, -0.4, 1.1, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        
        # Build full normalization bounds
        obs_dim = self.ROBOT_FEATS + self.MAX_OBJECTS * self.OBJ_FEATS
        self._obs_low = np.concatenate([
            self._obs_robot_low,
            np.tile(self._obs_obj_low, self.MAX_OBJECTS)
        ])
        self._obs_high = np.concatenate([
            self._obs_robot_high,
            np.tile(self._obs_obj_high, self.MAX_OBJECTS)
        ])
        
        # Observation space: normalized to [-1, 1]
        # Observation is [current_state, goal_state] concatenated
        observation_space = Box(
            low=-1.0, 
            high=1.0, 
            shape=(obs_dim * 2,), 
            dtype=np.float32
        )

        MujocoEnv.__init__(
            self, "simple_scene.xml", 50, observation_space=observation_space, **kwargs_filtered
        )
        
        # ---------------------------------------------------------------------------------------------------------------------
        print("========== ACTION SPACE DEBUG ==========")
        print("nu (number of actuators):", self.model.nu)
        print("Actuator names:")
        for i in range(self.model.nu):
            print(f"  {i}: {self.model.actuator(i).name}")
        print("\nCtrlrange:")
        print(self.model.actuator_ctrlrange)

        print("\nGym action_space:")
        print(self.action_space)
        print("========================================")

        #---------------------------------------------------------------------------------------------------------------------
        model = self.model                  # mujoco MjModel
        self.target_obj_id = 0              # Target object ID
        self.object_geom_ids = {}
        for obj_id in range(self.MAX_OBJECTS):
            name = f"object{obj_id}"
            try:
                self.object_geom_ids[obj_id] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                
            except Exception as e:
                print(f"Warning: Could not find geom names: {e}")
                self.object_geom_ids[obj_id] = None
                # self.finger_geom_ids = []

        self.finger_geom_ids = [
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_geom"),
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_geom"),
                ]

        self.ik = MinkIK(
            model=self.model,
            data=self.data,
            target_name="target0"
        )

        # self._lift_streak = 0            # (testing2)
        # self._lift_streak_required = 20   # K=20 = 1.0 sec consecutive steps to confirm lift (testing2)
        
        self._last_successful_ik_qpos = None

        # Print joint information, to check if the index is correct
        print("Number of joints:", model.njnt)
        for i in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            joint_adr = model.jnt_qposadr[i]
            print(f"Joint {i}: {joint_name}, qpos_addr: {joint_adr}")

        # in __init__
        self._ctrl_target = np.zeros(self.model.nu, dtype=np.float32)

        # per-actuator step sizes (delta per env step when action=1)
        # order matches actuators: base_pos, lift, arm, gripper  (all position actuators now)
        # Optimized for 75-step horizon: can traverse ~2x full range per episode
        self._delta_scale = np.array([
            0.015, # base_pos: 1.5cm/step (range [-0.4, 0.4] = 0.8m, ~53 steps full range)
            0.015, # lift position target increment (1.5cm/step, 38 steps for full range)
            0.015, # arm position target increment (1.5cm/step, 35 steps for full range)
            0.05,  # gripper (unchanged, 5cm/step)
        ], dtype=np.float32)[: self.model.nu]

        self.reset()
        #---------------------------------------------------------------------------------------------------------------

    def set_next_reset_object_pose(self, xyz):
        """
        Force object pose for the next reset only.
        Expected xyz in world coordinates; z defaults to table height if omitted.
        """
        arr = np.asarray(xyz, dtype=np.float64).reshape(-1)
        if arr.size < 2:
            raise ValueError(f"Expected object pose with at least x,y, got shape {arr.shape}")

        if arr.size < 3:
            arr = np.array([arr[0], arr[1], 0.75], dtype=np.float64)

        self._next_reset_object_xyz = arr[:3].copy()

    def _get_obs_internal(self):
        """Get observation without goal concatenation (for internal use)."""
        qpos = self.data.qpos

        # Robot state (5 dims): base_x, lift, arm_ext, gripper, is_grasped
        # Removed wrist_yaw, wrist_pitch, wrist_roll since they're always 0
        base_x      = qpos[IDX_BASE_X]
        lift        = qpos[IDX_LIFT]
        arm_ext     = qpos[IDX_ARM_L0] + qpos[IDX_ARM_L1] + qpos[IDX_ARM_L2] + qpos[IDX_ARM_L3]
        gripper     = qpos[IDX_GRIPPER]
        is_grasped  = self._get_grasp_flag()

        robot_state = np.array([
            base_x,
            lift, 
            arm_ext,
            gripper, 
            is_grasped
        ])

        # Object states (MAX_OBJECTS * 7 dims)
        # Only fill in num_objects, rest are zeros
        objects_state = np.zeros(self.MAX_OBJECTS * self.OBJ_FEATS, dtype=np.float64)
        
        for obj_id in self.objects:
            obj_name = f"object{obj_id}"
            try:
                obj_pos = self.get_body_com(obj_name)  # (3,) get the object position
                
                # Get object quaternion from qpos
                # Assuming objects are added sequentially in XML
                obj_qpos_start = IDX_OBJ_X + obj_id * 7  # Each object freejoint has 7 qpos
                obj_qw = qpos[obj_qpos_start + 3]
                obj_qx = qpos[obj_qpos_start + 4]
                obj_qy = qpos[obj_qpos_start + 5]
                obj_qz = qpos[obj_qpos_start + 6]
                
                # Fill in object's features at its actual object ID slot: [x, y, z, qw, qx, qy, qz] (position-first)
                obj_idx = obj_id * self.OBJ_FEATS  # Use actual object ID as slot index
                objects_state[obj_idx:obj_idx + 7] = [
                    obj_pos[0], obj_pos[1], obj_pos[2],  # position first
                    obj_qw, obj_qx, obj_qy, obj_qz        # then quaternion
                ]
            except:
                # If object doesn't exist, leave as zeros
                pass

        return np.concatenate([robot_state, objects_state])
    
    def _normalize_obs(self, obs):
        """Normalize observation to [-1, 1] range."""
        # Normalize robot state (except is_grasped which stays as boolean 0/1)
        robot_obs = obs[:self.ROBOT_FEATS]
        robot_low = self._obs_robot_low
        robot_high = self._obs_robot_high
        robot_range = robot_high - robot_low
        robot_range = np.where(robot_range < 1e-8, 1.0, robot_range)
        
        # Normalize first 4 features (base_x, lift, arm_ext, gripper)
        robot_clipped = np.clip(robot_obs[:4], robot_low[:4], robot_high[:4])
        robot_normalized = (robot_clipped - robot_low[:4]) / robot_range[:4] # scale to [0, 1]
        robot_normalized = 2.0 * robot_normalized - 1.0 # scale to [-1, 1]
        
        # Keep is_grasped (5th feature, index 4) as-is (0 or 1)
        is_grasped = robot_obs[4]
        
        # Concatenate normalized robot features with unnormalized is_grasped
        robot_normalized = np.concatenate([robot_normalized, [is_grasped]])
        
        # Normalize object states (only for existing objects, leave others as 0)
        objects_normalized = np.zeros(self.MAX_OBJECTS * self.OBJ_FEATS, dtype=np.float32)
        
        for obj_id in self.objects:
            obj_start = self.ROBOT_FEATS + obj_id * self.OBJ_FEATS  # Use actual object ID as slot index
            obj_end = obj_start + self.OBJ_FEATS
            obj_obs = obs[obj_start:obj_end]
            
            # Check if object exists (position is non-zero)
            if np.any(obj_obs[:3] != 0):  # Check x, y, z
                obj_range = self._obs_obj_high - self._obs_obj_low
                obj_range = np.where(obj_range < 1e-8, 1.0, obj_range)
                
                obj_clipped = np.clip(obj_obs, self._obs_obj_low, self._obs_obj_high)
                obj_normalized = (obj_clipped - self._obs_obj_low) / obj_range
                obj_normalized = 2.0 * obj_normalized - 1.0
                
                objects_normalized[obj_id * self.OBJ_FEATS:(obj_id + 1) * self.OBJ_FEATS] = obj_normalized
        
        return np.concatenate([robot_normalized, objects_normalized]).astype(np.float32)
    
    def _get_obs(self):
        """Get full observation with goal (SGCRL pattern)."""
        obs = self._get_obs_internal()
        # Normalize both current observation and goal
        obs_norm = self._normalize_obs(obs)
        goal_norm = self._normalize_obs(self._goal)
        # Return [obs, goal] concatenated
        return np.concatenate([obs_norm, goal_norm]).astype(np.float32)

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Reset robot base state with RANDOM x position within joint limits
        # Base range from stretch.xml: [-0.65, 0.65]
        BASE_X_MIN = -0.4
        BASE_X_MAX = 0.4
        qpos[IDX_BASE_X] = self.np_random.uniform(BASE_X_MIN, BASE_X_MAX)
        
        # Reset robot joint positions
        qpos[IDX_LIFT]   = 0.75  # Lift body at 0.2+0.75=0.95m, allows reaching table at 0.72m
        qpos[IDX_ARM_L0] = 0.0
        qpos[IDX_ARM_L1] = 0.0
        qpos[IDX_ARM_L2] = 0.0
        qpos[IDX_ARM_L3] = 0.0
        # Fix gripper orientation (wrist joints always at 0)
        qpos[IDX_WRIST_YAW]   = 0.0
        qpos[IDX_WRIST_PITCH] = 0.0
        qpos[IDX_WRIST_ROLL]  = 0.0
        # qpos[IDX_GRIPPER] = 0.0
        qpos[IDX_GRIPPER]     = self._obs_robot_high[3]  # Use max gripper opening

        self._ctrl = self.data.ctrl.copy()  # shape (nu,)

        # Reset object(s) position

        TABLE_X_CENTER = 0.0
        TABLE_Y_CENTER = -0.55
        TABLE_X_HALF   = 0.3   # < 0.4, leave margin
        TABLE_Y_HALF   = 0.1    # < 0.15, leave margin
        forced_xyz = self._next_reset_object_xyz
        self._next_reset_object_xyz = None

        for obj_id in self.objects:
            if forced_xyz is not None and obj_id == self.target_obj_id:
                obj_xy = np.array([forced_xyz[0], forced_xyz[1]], dtype=np.float64)
                obj_z = float(forced_xyz[2])
            else:
                obj_xy = self.np_random.uniform(
                    low=np.array([TABLE_X_CENTER - TABLE_X_HALF, TABLE_Y_CENTER - TABLE_Y_HALF]),
                    high=np.array([TABLE_X_CENTER + TABLE_X_HALF, TABLE_Y_CENTER + TABLE_Y_HALF]),
                )
                obj_z = 0.75
            
            obj_qpos_start = IDX_OBJ_X + obj_id * 7
            qpos[obj_qpos_start]     = obj_xy[0]  # x
            qpos[obj_qpos_start + 1] = obj_xy[1]  # y
            qpos[obj_qpos_start + 2] = obj_z      # z
            qpos[obj_qpos_start + 3] = 1.0        # qw
            qpos[obj_qpos_start + 4] = 0.0        # qx
            qpos[obj_qpos_start + 5] = 0.0        # qy
            qpos[obj_qpos_start + 6] = 0.0        # qz

        qvel[:] = 0.0
        self.set_state(qpos, qvel) # set it into mujoco state

        mujoco.mj_forward(self.model, self.data)               # update the sim state
        self.ee_quat_ref = self._site_quat("ee_site").copy()   # Gets the end-effector orientation at the reset state
        
        # ---- Generate goal EVERY episode ----
        current_obs = self._get_obs_internal() # get current observation (without goal)

        print(f"[GOAL GENERATION DEBUG] _fixed_start_end = {self._fixed_start_end}")

        if forced_xyz is not None:
            # Goal image comes from presampled pkl context (in ContextualEnv),
            # so skip IK here to avoid changing the forced object pose.
            print("[GOAL] Forced reset pose detected, skipping _compute_goal_with_ik()")
            self._goal = np.asarray(current_obs, dtype=np.float32)
            self._goal_qpos = self.data.qpos.copy()
            self._goal_qvel = self.data.qvel.copy()

        elif self._fixed_start_end is not None:
            # Fixed goal (for debugging / ablation)
            print("[GOAL] Using fixed goal from _fixed_start_end parameter")
            self._goal = np.asarray(self._fixed_start_end[1], dtype=np.float32)

        else:
            print("[GOAL] Computing goal with IK solver (per-episode)")

            slot = self._slot_of_obj(self.target_obj_id)
            base = self.ROBOT_FEATS + slot * self.OBJ_FEATS
            obj_x = current_obs[base + 0]
            obj_y = current_obs[base + 1]
            obj_z = current_obs[base + 2]

            # obj_z_goal = 0.9  # table 0.75 + 0.15 (pick task)
            obj_z_goal = obj_z  # Keep at original height (align gripper to object)

            print(f"[GOAL] Original object position: ({obj_x:.3f}, {obj_y:.3f}, {obj_z:.3f})")
            print(f"[GOAL] Target goal position:   ({obj_x:.3f}, {obj_y:.3f}, {obj_z_goal:.3f})")
            print(f"[GOAL] Goal: EE aligned and in front of object (pre-grasp)")

            self._goal = self._compute_goal_with_ik(self.target_obj_id, obj_x, obj_y, obj_z_goal)    # set up goal observation
            # self._goal = self._compute_goal_with_ik(obj_x, obj_y, obj_z_goal)

        # self._lift_streak = 0  # reset lift streak counter (testing2)
        
        # Initialize control targets to valid values within actuator ranges
        # All actuators are now position actuators — initialize to current qpos
        self._ctrl_target[0] = qpos[IDX_BASE_X]    # base position
        self._ctrl_target[1] = qpos[IDX_LIFT]      # lift position
        self._ctrl_target[2] = qpos[IDX_ARM_L0] + qpos[IDX_ARM_L1] + qpos[IDX_ARM_L2] + qpos[IDX_ARM_L3]  # arm extension
        self._ctrl_target[3] = qpos[IDX_GRIPPER]       # gripper
        
        # Apply the control targets
        self.data.ctrl[:] = self._ctrl_target.copy()


        return self._get_obs()

    def _slot_of_obj(self, obj_id: int) -> int:  # decide which slot an object is in
        return self.objects.index(obj_id)  

    def step(self, a):
        # ── Direct qpos override (linear interpolation policy) ───────────────
        # DISABLED: this teleport branch produces transitions under different
        # dynamics than the normal mj_step path used during RL training.
        # Keeping the code for reference/debugging only — never fires in practice
        # because _allow_teleport defaults to False.
        _dq = getattr(self, '_direct_qpos', None)
        if _dq is not None and getattr(self, '_allow_teleport', False):
            self._direct_qpos = None  # consume it

            base_x, lift, arm_ext, gripper = float(_dq[0]), float(_dq[1]), float(_dq[2]), float(_dq[3])
            arm_per_joint = arm_ext / 4.0

            # Teleport to desired position
            self.data.qpos[IDX_BASE_X]   = base_x
            self.data.qpos[IDX_LIFT]     = lift
            self.data.qpos[IDX_ARM_L0]   = arm_per_joint
            self.data.qpos[IDX_ARM_L1]   = arm_per_joint
            self.data.qpos[IDX_ARM_L2]   = arm_per_joint
            self.data.qpos[IDX_ARM_L3]   = arm_per_joint
            self.data.qpos[IDX_GRIPPER]  = gripper
            # Finger joints coupled to gripper slide: finger_pos = 10 * gripper_slide
            self.data.qpos[IDX_GRIPPER_LEFT]  = 10.0 * gripper
            self.data.qpos[IDX_GRIPPER_RIGHT] = 10.0 * gripper
            self.data.qvel[:] = 0.0  # zero velocity — pure position control

            # Update ctrl to match so the actuators don't fight back
            self._ctrl_target[0] = base_x     # base_pos = target position
            self._ctrl_target[1] = lift
            self._ctrl_target[2] = arm_ext
            self._ctrl_target[3] = gripper
            self.data.ctrl[:] = self._ctrl_target.copy()

            mujoco.mj_forward(self.model, self.data)  # update kinematics (no physics step)

            if self.render_mode == "human":
                self.render()

            full_obs = self._get_obs()
            obs_dim = self.ROBOT_FEATS + self.MAX_OBJECTS * self.OBJ_FEATS
            current_obs = full_obs[:obs_dim]
            obj_pos_wc = self.get_body_com(f"object{self.target_obj_id}").copy()
            ee_site_id = self.model.site("ee_site").id
            ee_pos = self.data.site_xpos[ee_site_id].copy()
            dx = ee_pos[0] - obj_pos_wc[0]
            dy = ee_pos[1] - obj_pos_wc[1]
            X_TOL, FRONT_MARGIN = 0.03, 0.15
            x_aligned = abs(dx) <= X_TOL
            in_front = (dy <= FRONT_MARGIN) and (dy > 0.0)
            success = bool(x_aligned and in_front)
            info = {"success": success, "dx": float(dx), "dy": float(dy),
                    "x_aligned": bool(x_aligned), "in_front": bool(in_front)}
            return full_obs, float(success), False, False, info
        # ─────────────────────────────────────────────────────────────────────

        # Normal action → delta → filter path (for RL agent use)
        a_raw = np.asarray(a).copy()
        print(f"[step] 0. a_raw (from policy): base={a_raw[0]:.6f}, lift={a_raw[1]:.6f}, arm={a_raw[2]:.6f}, gripper={a_raw[3]:.6f}")
        a = np.copy(a)
        a = np.clip(a, -1.0, 1.0)
        print(f"[step] 1. a (after clip/before override): base={a[0]:.6f}, lift={a[1]:.6f}, arm={a[2]:.6f}, gripper={a[3]:.6f}")
        a[3] = 1.0

        # All actuators are now position actuators — accumulate uniformly
        self._ctrl_target += a * self._delta_scale
        print(f"[step] 2. _ctrl_target(before clip): base_pos={self._ctrl_target[0]:.6f}, lift={self._ctrl_target[1]:.6f}, arm={self._ctrl_target[2]:.6f}, gripper={self._ctrl_target[3]:.6f}")

        low, high = self.model.actuator_ctrlrange.T
        self._ctrl_target = np.clip(self._ctrl_target, low, high)

        print(f"[step] 1. a (after clip/override): base={a[0]:.6f}, lift={a[1]:.6f}, arm={a[2]:.6f}, gripper={a[3]:.6f}")
        print(f"[step] 2. _ctrl_target: base_pos={self._ctrl_target[0]:.6f}, lift={self._ctrl_target[1]:.6f}, arm={self._ctrl_target[2]:.6f}, gripper={self._ctrl_target[3]:.6f}")

        # alpha=1.0 means ctrl tracks _ctrl_target directly (no lag).
        # The position actuators already have their own PD damping (gainprm/biasprm
        # in stretch.xml) so they are stable without an extra low-pass filter.
        # Keeping alpha < 1 here would cause demo dynamics to diverge from what
        # the RL policy expects, and wastes steps ramping up at episode start.
        # alpha = 1.0
        # self.data.ctrl[:] = (1 - alpha) * self.data.ctrl[:] + alpha * self._ctrl_target
        self.data.ctrl[:] = self._ctrl_target

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        # mujoco.mj_forward(self.model, self.data)

        # ── Robot pose vs goal pose debug ──────────────────────────────────
        _qp = self.data.qpos
        _arm_now = _qp[IDX_ARM_L0] + _qp[IDX_ARM_L1] + _qp[IDX_ARM_L2] + _qp[IDX_ARM_L3]
        print(f"[step] 3. robot_qpos : base={_qp[IDX_BASE_X]:.4f}, lift={_qp[IDX_LIFT]:.4f}, "
              f"arm={_arm_now:.4f}, gripper={_qp[IDX_GRIPPER]:.4f}")
        if hasattr(self, '_goal_qpos'):
            _gq = self._goal_qpos
            _arm_goal = _gq[IDX_ARM_L0] + _gq[IDX_ARM_L1] + _gq[IDX_ARM_L2] + _gq[IDX_ARM_L3]
            print(f"[step] 4. goal_qpos  : base={_gq[IDX_BASE_X]:.4f}, lift={_gq[IDX_LIFT]:.4f}, "
                  f"arm={_arm_goal:.4f}, gripper={_gq[IDX_GRIPPER]:.4f}")
            print(f"[step] 5. delta      : base={_gq[IDX_BASE_X]-_qp[IDX_BASE_X]:+.4f}, "
                  f"lift={_gq[IDX_LIFT]-_qp[IDX_LIFT]:+.4f}, "
                  f"arm={_arm_goal-_arm_now:+.4f}, "
                  f"gripper={_gq[IDX_GRIPPER]-_qp[IDX_GRIPPER]:+.4f}")
        # ───────────────────────────────────────────────────────────────────

       # self.do_simulation(np.clip(self._ctrl, low, high), self.frame_skip)  # take action
        
       
        # self.data.qpos[IDX_WRIST_YAW] = 0.0
        # self.data.qpos[IDX_WRIST_PITCH] = 0.0
        # self.data.qpos[IDX_WRIST_ROLL] = 0.0
        # self.data.qpos[IDX_GRIPPER] = 0.05
        # mujoco.mj_forward(self.model, self.data)
        
        if self.render_mode == "human":
            self.render()
            # This prints every step when render_mode=="human"
            # print(f"[RENDER] Step {self.data.time:.2f}s - Rendering frame")  # Debug print

        full_obs = self._get_obs()
        
        # Extract current observation (first half of full_obs)
        obs_dim = self.ROBOT_FEATS + self.MAX_OBJECTS * self.OBJ_FEATS
        current_obs = full_obs[:obs_dim]  # ONLY current state
        # goal_obs    = full_obs[obs_dim:] # for (testing1)
        
        # Extract first object's position from current observation
        # Object state format: [x, y, z, qw, qx, qy, qz] (position-first)
        slot = self._slot_of_obj(self.target_obj_id)
        base = self.ROBOT_FEATS + slot * self.OBJ_FEATS
        # obj_x = current_obs[base + 0]  # Not used for success check
        # obj_y = current_obs[base + 1]  # Not used for success check
        # obj_z = current_obs[base + 2]  # Not used for success check
        is_grasped = current_obs[self.ROBOT_FEATS - 1]  # is_grasped is last robot feature
        # base_x = current_obs[0]      # robot base_x (testing1)
        # goal_base_x = goal_obs[0]    # goal base_x  (testing1)
        
        # obj_pos = np.array([obj_x, obj_y, obj_z])  # Removed: using fresh obj_pos_wc instead

        obj_pos_wc = self.get_body_com(f"object{self.target_obj_id}").copy()  # (align gripper to object) get updated object position from sim

        ee_site_id = self.model.site("ee_site").id    # Get End-Effector position (align gripper to object)
        ee_pos = self.data.site_xpos[ee_site_id].copy()  # (align gripper to object) Get updated EE position from sim
        dx = ee_pos[0] - obj_pos_wc[0]   # x alignment。(align gripper to object)
        dy = ee_pos[1] - obj_pos_wc[1]   # y relative (dy is positive) (front/back) (align gripper to object)
        
        # # ---------------------------------------(align gripper to object)-------------------------------------
        X_TOL = 0.03          # 3cm
        FRONT_MARGIN = 0.15   # 15cm

        x_aligned = abs(dx) <= X_TOL 
        in_front = (dy <= FRONT_MARGIN) and (dy > 0.0)
        # Threshold for alignment (e.g., 0.05m = 5cm)
        success = bool(x_aligned and in_front)
        r = float(success)

        info = {
            "success": success,
            "dx": float(dx),
            "dy": float(dy),
            "x_aligned": bool(x_aligned),
            "in_front": bool(in_front),
        }
        terminated = False
        truncated = False
        # ---------------------------------------(align gripper to object)-------------------------------------
        #---------------------------------------(testing1)-------------------------------------
        # X_TOL = 0.05                 # (testing1)
        # success = bool(abs(base_x - goal_base_x) < X_TOL) # (testing1)
        # r = float(success) # (testing1)
        # info = {"success": success, "base_x": float(base_x), "goal_base_x": float(goal_base_x)} # (testing1)
        # terminated = False # (testing1)
        # truncated = False  # (testing1)

        # --------update lift streak (is_grasped is 0/1) (testing2)-----------------------------------
        # --- lift streak (anti-bounce) ---
        # LIFT_THRESH = 0.85
        # if obj_pos[2] > LIFT_THRESH:
        #     self._lift_streak += 1
        # else:
        #     self._lift_streak = 0

        # stable_lift = (self._lift_streak >= self._lift_streak_required)

        # # Phase 1: only require stable lift
        # success = bool(stable_lift)

        # # Phase 2: stable lift + grasp
        # # success = bool(stable_lift and (is_grasped > 0.5))

        # r = float(success)

        # info = {
        #     "success": success,
        #     "obj_pos": obj_pos,
        #     "is_grasped": float(is_grasped),
        #     "lift_streak": int(self._lift_streak),
        #     "stable_lift": bool(stable_lift),
        # }
        # terminated = False
        # truncated = False
        
        #---------------------------------------(testing2)-------------------------------------
        #---------------------------------------(pick task)-------------------------------------
        # Use fresh object position from simulation (after mj_step)
        # obj_pos_wc = self.get_body_com(f"object{self.target_obj_id}").copy()
        
        # LIFT_THRESH = 0.85  # 10cm above table (table at 0.72m, objects start at 0.75m)
        
        # # Success = object lifted AND grasped by robot
        # success = bool(
        #     (obj_pos_wc[2] > LIFT_THRESH) and  # Object height check
        #     (is_grasped > 0.5)                   # Grasp check (both fingers touching)
        # )

        # # SGCRL pattern: episodes don't terminate on success, only on max_episode_steps
        # terminated = False
        # truncated = False

        # r = float(success)

        # info = {
        #     "success": success,
        #     "obj_pos": obj_pos_wc,  # Report fresh position
        #     "obj_height": float(obj_pos_wc[2]),
        #     "is_grasped": float(is_grasped),
        # }
        #---------------------------------------(pick task)-------------------------------------

        return full_obs, r, terminated, truncated, info
    
    def _get_grasp_flag(self) -> float:
        object_geom_id = self.object_geom_ids.get(self.target_obj_id, None)
        if object_geom_id is None or len(self.finger_geom_ids) < 2:
            return 0.0

        left_id, right_id = self.finger_geom_ids[0], self.finger_geom_ids[1]
        left_touch = False
        right_touch = False

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2

            if (g1 == object_geom_id and g2 == left_id) or (g2 == object_geom_id and g1 == left_id):
                left_touch = True
            if (g1 == object_geom_id and g2 == right_id) or (g2 == object_geom_id and g1 == right_id):
                right_touch = True

            if left_touch and right_touch:
                return 1.0

        return 0.0

    def _safe_render(self, n=1):
    # In launchpad/acme actor workers, self.spec is often None -> render() asserts.
        if getattr(self, "spec", None) is None:
            return
        for _ in range(n):
            try:
                self.render()
            except AssertionError:
                return
        
    def _site_quat(self, site_name: str) -> np.ndarray:
            sid = self.model.site(site_name).id
            mat = self.data.site_xmat[sid].copy()          # (9,) 3*3
            quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quat, mat)                 # mat is 9 floats
            return quat

    def get_contextual_diagnostics(self, paths, contexts):
        """
        Called by ContextualEnv.get_diagnostics() after each epoch.

        Roboverse reads state_observation/state_desired_goal from path observations
        and recomputes metrics from scratch. We cannot do the same because this is
        an image-based env (use_image=True): path observations only contain
        image_observation/image_desired_goal, not state vectors.

        Instead we read the pre-computed values from env_infos stored during step(),
        and also add an EE-to-object distance metric (equivalent to roboverse's
        distance metrics). The format matches roboverse exactly:
          - state_desired_goal/final/<metric>: last-step value per episode
          - state_desired_goal/<metric>:       all-step average across episodes
        """
        from collections import OrderedDict
        from multiworld.envs.env_util import create_stats_ordered_dict

        diagnostics = OrderedDict()
        goal_key = 'state_desired_goal'

        # success/boolean flags and scalar distances from env_infos
        # 'ee_to_obj_distance' = sqrt(dx^2 + dy^2), analogous to roboverse distance metrics
        scalar_metrics = {
            'success':           'align_success',
            'x_aligned':         'x_aligned',
            'in_front':          'in_front',
        }
        distance_metrics = {
            'dx':  'dx',
            'dy':  'dy',
        }

        final_vals = {k: [] for k in list(scalar_metrics) + list(distance_metrics) + ['ee_to_obj_distance']}
        all_vals   = {k: [] for k in list(scalar_metrics) + list(distance_metrics) + ['ee_to_obj_distance']}

        for path in paths:
            env_infos = path.get('env_infos', [])
            if not env_infos:
                continue
            for info in env_infos:
                for k in list(scalar_metrics) + list(distance_metrics):
                    if k in info:
                        all_vals[k].append(float(info[k]))
                if 'dx' in info and 'dy' in info:
                    dist = float(np.sqrt(info['dx']**2 + info['dy']**2))
                    all_vals['ee_to_obj_distance'].append(dist)

            last = env_infos[-1]
            for k in list(scalar_metrics) + list(distance_metrics):
                if k in last:
                    final_vals[k].append(float(last[k]))
            if 'dx' in last and 'dy' in last:
                final_vals['ee_to_obj_distance'].append(
                    float(np.sqrt(last['dx']**2 + last['dy']**2)))

        # Report final-step success/flag metrics
        for info_key, display_name in scalar_metrics.items():
            if final_vals[info_key]:
                diagnostics.update(create_stats_ordered_dict(
                    goal_key + f'/final/{display_name}',
                    np.array(final_vals[info_key])))
            if all_vals[info_key]:
                diagnostics.update(create_stats_ordered_dict(
                    goal_key + f'/{display_name}',
                    np.array(all_vals[info_key])))

        # Report final-step distance metrics (dx, dy, ee_to_obj_distance)
        for info_key, display_name in distance_metrics.items():
            if final_vals[info_key]:
                diagnostics.update(create_stats_ordered_dict(
                    goal_key + f'/final/{display_name}_distance',
                    np.array(final_vals[info_key])))
            if all_vals[info_key]:
                diagnostics.update(create_stats_ordered_dict(
                    goal_key + f'/{display_name}_distance',
                    np.array(all_vals[info_key])))

        if final_vals['ee_to_obj_distance']:
            diagnostics.update(create_stats_ordered_dict(
                goal_key + '/final/ee_to_obj_distance',
                np.array(final_vals['ee_to_obj_distance'])))
        if all_vals['ee_to_obj_distance']:
            diagnostics.update(create_stats_ordered_dict(
                goal_key + '/ee_to_obj_distance',
                np.array(all_vals['ee_to_obj_distance'])))

        return diagnostics

# -------------------------pick task-----------------------------------------------------------------
    # def _compute_goal_with_ik(self, obj_id, obj_x, obj_y, obj_z_goal):
    #     """Compute goal observation using IK solver to place end-effector at target object position."""
    #     # 1. Save state
    #     qpos0 = self.data.qpos.copy()
    #     qvel0 = self.data.qvel.copy()
        
    #     # 2. Set mocap target
    #     mocap_id = int(self.model.body_mocapid[
    #         mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target0")
    #     ])

    #     self.data.mocap_pos[mocap_id] = np.array([obj_x, obj_y, obj_z_goal]) # move the mocap body to the target position
    #     self.data.mocap_quat[mocap_id] = self.ee_quat_ref.copy() # set the mocap orientation to the reference end-effector orientation
    #     mujoco.mj_forward(self.model, self.data)
        
    #     # 3. Update IK and solve
    #     self.ik.update_configuration(self.data.qpos)
    #     success = self.ik.converge_ik(dt=0.01)  # move te ee to the target position

    #     # Apply solution
    #     self.data.qpos[:] = self.ik.configuration.q
    #     mujoco.mj_forward(self.model, self.data)

    #     # Check actual error
    #     ee_site_id = self.model.site("ee_site").id
    #     ee_pos = self.data.site_xpos[ee_site_id].copy()
    #     target_pos = self.data.mocap_pos[mocap_id].copy()
    #     ik_error = np.linalg.norm(target_pos - ee_pos)

    #     print(f"[IK] MinkIK success={success}, actual EE error={ik_error:.6f}m")

    #     # 4. Check BOTH conditions: IK convergence AND actual error
    #     IK_ERROR_THRESHOLD = 0.01  # 1cm tolerance
        
    #     if not success or ik_error > IK_ERROR_THRESHOLD:
    #         # IK failed OR error too large
    #         if not success:
    #             print(f"[IK] REJECTED - IK did not converge")
    #         if ik_error > IK_ERROR_THRESHOLD:
    #             print(f"[IK] REJECTED - error {ik_error:.4f}m > {IK_ERROR_THRESHOLD}m")
            
    #         # 5. Resample if failed - sample from ENTIRE table
    #         self.set_state(qpos0, qvel0)
    #         # self.data.mocap_pos[:] = mocap_pos0
    #         mujoco.mj_forward(self.model, self.data)
            
    #         # Resample from entire table workspace
    #         TABLE_X_CENTER = 0.0
    #         TABLE_Y_CENTER = -0.55
    #         TABLE_X_HALF   = 0.3
    #         TABLE_Y_HALF   = 0.1
            
    #         obj_xy_new = self.np_random.uniform(
    #             low=np.array([TABLE_X_CENTER - TABLE_X_HALF, TABLE_Y_CENTER - TABLE_Y_HALF]),
    #             high=np.array([TABLE_X_CENTER + TABLE_X_HALF, TABLE_Y_CENTER + TABLE_Y_HALF]),
    #         )
    #         obj_x_new = obj_xy_new[0]
    #         obj_y_new = obj_xy_new[1]
            
    #         if not hasattr(self, '_ik_attempts'):
    #             self._ik_attempts = 0
    #         self._ik_attempts += 1
            
    #         print(f"[IK] Retry attempt {self._ik_attempts} with new position ({obj_x_new:.3f}, {obj_y_new:.3f})")
    #         result = self._compute_goal_with_ik(obj_id, obj_x_new, obj_y_new, obj_z_goal)
    #         self._ik_attempts = 0
    #         return result

    #     # If we reach here: IK succeeded AND error is acceptable
    #     print(f"[IK] ACCEPTED - success={success}, error={ik_error:.6f}m")
    #     self._ik_attempts = 0  # Reset on success

    #     self._last_successful_ik_qpos = self.data.qpos.copy()  # Cache successful IK solution
    #     print("[IK] Cached successful IK solution for future fallbacks")

    #     # Continue with goal generation (no need to re-apply IK solution)
        
    #     # 6. Move object to goal position
    #     obj_qpos_start = IDX_OBJ_X + obj_id * 7
    #     self.data.qpos[obj_qpos_start + 0] = obj_x
    #     self.data.qpos[obj_qpos_start + 1] = obj_y
    #     self.data.qpos[obj_qpos_start + 2] = obj_z_goal

    #     # 7. Enforce wrist orientation
    #     self.data.qpos[IDX_WRIST_YAW] = 0.0
    #     self.data.qpos[IDX_WRIST_PITCH] = 0.0
    #     self.data.qpos[IDX_WRIST_ROLL] = 0.0
        
    #     # Close gripper
    #     # self.data.qpos[IDX_GRIPPER] = -0.1
    #     mujoco.mj_forward(self.model, self.data)

    #     # 8. Get observation (unnormalized - will be normalized when used)
    #     goal_obs = self._get_obs_internal()
    #     goal_obs[self.ROBOT_FEATS - 1] = 1.0  # Force is_grasped flag
        
    #     # Debug output
    #     ee_pos_final = self.data.site_xpos[ee_site_id].copy()
    #     print("=== IK Debug ===")
    #     print(f"Target (mocap) pos : {target_pos}")
    #     print(f"Gripper (EE) pos   : {ee_pos_final}")
    #     print(f"EE error (L2)      : {np.linalg.norm(target_pos - ee_pos_final):.6f}m")
    #     print("================")

    #     print("=== IK GOAL OBS (unnormalized) ===")
    #     print("shape:", goal_obs.shape)
    #     print("Robot state:", goal_obs[:self.ROBOT_FEATS])
    #     print("ee_site pos:", self.data.site_xpos[self.model.site("ee_site").id])
    #     print("===================")

    #     # 9. Restore state
    #     self.set_state(qpos0, qvel0)
    #     mujoco.mj_forward(self.model, self.data)

    #     return goal_obs

# -------------------------pick task---------------------------------------------------------------
# -----------------------align gripper to object with IK-------------------------------------------
    def _compute_goal_with_ik(self, obj_id, obj_x, obj_y, obj_z_goal):
        """Compute goal observation using IK solver to place end-effector at target object position."""
        # 1. Save state
        qpos0 = self.data.qpos.copy()
        qvel0 = self.data.qvel.copy()
        
        # 2a. IMPORTANT: Move object to target position (obj_x, obj_y, obj_z_goal)
        # This is critical when IK retry samples a new position - object must match
        obj_qpos_start = IDX_OBJ_X + obj_id * 7
        self.data.qpos[obj_qpos_start + 0] = obj_x       # x position
        self.data.qpos[obj_qpos_start + 1] = obj_y       # y position
        self.data.qpos[obj_qpos_start + 2] = obj_z_goal  # z position
        # Keep quaternion (indices 3-6) unchanged - object doesn't rotate
        
        # 2b. Set mocap target - position EE in front of object (not at lifted position)
        # Target: same X as object, slightly in front in Y (negative Y direction)
        # APPROACH_OFFSET_Y = -0.10  # 10cm in front of object (negative Y)
        APPROACH_OFFSET_Y = 0.12  # 15cm in front of object (negative Y) - Centers goal in success region [-0.05, 0.0)
        target_x = obj_x
        # target_y = obj_y + APPROACH_OFFSET_Y  # Move EE in front
        target_y = obj_y + APPROACH_OFFSET_Y
        target_z = obj_z_goal  # Keep at table height (0.75m typically)
        
        mocap_id = int(self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target0")
        ])

        self.data.mocap_pos[mocap_id] = np.array([target_x, target_y, target_z])
        self.data.mocap_quat[mocap_id] = self.ee_quat_ref.copy()
        mujoco.mj_forward(self.model, self.data)
        
        # 3. Update IK and solve
        self.ik.update_configuration(self.data.qpos)
        success = self.ik.converge_ik(dt=0.01)  # move te ee to the target position

        # Apply solution
        self.data.qpos[:] = self.ik.configuration.q
        mujoco.mj_forward(self.model, self.data)

        # Check actual error
        ee_site_id = self.model.site("ee_site").id
        ee_pos = self.data.site_xpos[ee_site_id].copy()
        target_pos = self.data.mocap_pos[mocap_id].copy()
        ik_error = np.linalg.norm(target_pos - ee_pos)

        print(f"[IK] MinkIK success={success}, actual EE error={ik_error:.6f}m")

        # 4. Check BOTH conditions: IK convergence AND actual error
        IK_ERROR_THRESHOLD = 0.01  # 1cm tolerance
        
        if not success or ik_error > IK_ERROR_THRESHOLD:
            # IK failed OR error too large
            if not success:
                print(f"[IK] REJECTED - IK did not converge")
            if ik_error > IK_ERROR_THRESHOLD:
                print(f"[IK] REJECTED - error {ik_error:.4f}m > {IK_ERROR_THRESHOLD}m")
            
            # 5. Resample if failed - sample from ENTIRE table
            self.set_state(qpos0, qvel0)
            # self.data.mocap_pos[:] = mocap_pos0
            mujoco.mj_forward(self.model, self.data)
            
            # Resample from entire table workspace
            TABLE_X_CENTER = 0.0
            TABLE_Y_CENTER = -0.55
            TABLE_X_HALF   = 0.3
            TABLE_Y_HALF   = 0.1
            
            obj_xy_new = self.np_random.uniform(
                low=np.array([TABLE_X_CENTER - TABLE_X_HALF, TABLE_Y_CENTER - TABLE_Y_HALF]),
                high=np.array([TABLE_X_CENTER + TABLE_X_HALF, TABLE_Y_CENTER + TABLE_Y_HALF]),
            )
            obj_x_new = obj_xy_new[0]
            obj_y_new = obj_xy_new[1]
            
            if not hasattr(self, '_ik_attempts'):
                self._ik_attempts = 0
            self._ik_attempts += 1
            
            print(f"[IK] Retry attempt {self._ik_attempts} with new position ({obj_x_new:.3f}, {obj_y_new:.3f})")
            
            # CRITICAL FIX: If we resample the object position for the goal, we MUST update the 
            # environment's object position to match! Otherwise goal object != obs object.
            # qpos0 is the saved state that we will restore to at the end.
            # We need to update qpos0 with the new object position so that when we restore,
            # the object moves to the valid position.
            obj_qpos_start = IDX_OBJ_X + obj_id * 7
            qpos0[obj_qpos_start + 0] = obj_x_new
            qpos0[obj_qpos_start + 1] = obj_y_new
            
            # Also update the current simulation state before recursive call, 
            # so the recursive call starts with the correct object pos
            self.data.qpos[obj_qpos_start + 0] = obj_x_new
            self.data.qpos[obj_qpos_start + 1] = obj_y_new
            
            result = self._compute_goal_with_ik(obj_id, obj_x_new, obj_y_new, obj_z_goal)
            self._ik_attempts = 0
            return result

        # If we reach here: IK succeeded AND error is acceptable
        print(f"[IK] ACCEPTED - success={success}, error={ik_error:.6f}m")
        self._ik_attempts = 0  # Reset on success

        self._last_successful_ik_qpos = self.data.qpos.copy()  # Cache successful IK solution
        print("[IK] Cached successful IK solution for future fallbacks")

        # Continue with goal generation (no need to re-apply IK solution)
        
        # 6. Keep object at ORIGINAL position (not lifted) - goal is pre-grasp alignment
        obj_qpos_start = IDX_OBJ_X + obj_id * 7
        # Object stays at original table position - DO NOT move it
        # (qpos already has object at starting position from reset)

        # 7. Enforce wrist orientation and gripper opening
        # IMPORTANT: Do this AFTER IK solution is applied (line 743-745)
        self.data.qpos[IDX_WRIST_YAW] = 0.0
        self.data.qpos[IDX_WRIST_PITCH] = 0.0
        self.data.qpos[IDX_WRIST_ROLL] = 0.0
        
        # Keep gripper FULLY OPEN (not grasped yet, pre-grasp position)
        # Gripper slide range: [-0.02, 0.04]
        gripper_slide_open = self._obs_robot_high[3]  # 0.04 (max opening)
        self.data.qpos[IDX_GRIPPER] = gripper_slide_open
        
        # Finger joints are coupled to gripper slide via equality constraint:
        # finger_pos = 10 * gripper_slide (from polycoef="0 10 0 0 0" in XML)
        # When gripper_slide = 0.04, fingers should be at 0.4
        # Finger range: [-0.6, 0.6]
        finger_open_pos = 10.0 * gripper_slide_open  # 0.4 when slide = 0.04
        self.data.qpos[IDX_GRIPPER_LEFT] = finger_open_pos    # Open left finger
        self.data.qpos[IDX_GRIPPER_RIGHT] = finger_open_pos   # Open right finger
        
        # Forward kinematics to update the visual state with open gripper
        mujoco.mj_forward(self.model, self.data)

        # 8. Get observation (unnormalized - will be normalized when used)
        goal_obs = self._get_obs_internal()
        # is_grasped = 0.0 (not grasping yet, just aligned in front)
        goal_obs[self.ROBOT_FEATS - 1] = 0.0
        
        # Debug output
        ee_pos_final = self.data.site_xpos[ee_site_id].copy()
        print("=== IK DEBUG ===")
        print(f"Target (mocap) pos : {target_pos}")
        print(f"Gripper (EE) pos   : {ee_pos_final}")
        print(f"EE error (L2)      : {np.linalg.norm(target_pos - ee_pos_final):.6f}m")
        print("================")

        print("=== IK GOAL OBS (unnormalized) ===")
        print("shape:", goal_obs.shape)
        print(goal_obs)
        print("ee_site pos:", self.data.site_xpos[self.model.site("ee_site").id])
        print("===================================")
        
        # Normalize and print for debugging
        goal_obs_normalized = self._normalize_obs(goal_obs)
        print("=== IK GOAL OBS (normalized) ===")
        print("shape:", goal_obs_normalized.shape)
        print(goal_obs_normalized)
        print("Value range: [{:.3f}, {:.3f}]".format(goal_obs_normalized.min(), goal_obs_normalized.max()))
        print("=================================")

        # Store goal qpos for image rendering (before restoring state)
        self._goal_qpos = self.data.qpos.copy()
        self._goal_qvel = self.data.qvel.copy()
        # Store mocap position for goal rendering (red dot should stay with object)
        self._goal_mocap_pos = self.data.mocap_pos[mocap_id].copy()

        # 9. Restore state
        self.set_state(qpos0, qvel0)
        mujoco.mj_forward(self.model, self.data)

        return goal_obs
# -----------------------align gripper to object with IK-------------------------------------------
