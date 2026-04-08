import os
import glob

import gym
import numpy as np
from gym import spaces as gym_spaces

from absl import app
from absl import flags

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants
from rlkit.networks.gaussian_policy import GaussianCNNPolicy

from rlkit.learning.stable_contrastive_rl import stable_contrastive_rl_experiment
from rlkit.learning.stable_contrastive_rl import process_args
from rlkit.utils import arg_util
from rlkit.utils.logging import logger as logging


def _no_op_diagnostics(paths, contexts):
    return {}


class GymnasiumToGymWrapper(gym.Env):
    """
    Adapts a Gymnasium (5-tuple step) env to old gym (4-tuple step) API.
    Also converts gymnasium spaces to gym spaces so ClipAction works.
    Also adds get_image() so EnvRenderer can render from the env.
    Must be defined at module level so torch.save() can pickle it.
    """
    metadata = {}

    def __init__(self, gymnasium_env):
        self._env = gymnasium_env
        self.observation_space = self._convert_space(gymnasium_env.observation_space)
        self.action_space = self._convert_space(gymnasium_env.action_space)

        # Prefer diagnostics from the underlying env if it defines them,
        # otherwise fall back to no-op.
        underlying = getattr(gymnasium_env, 'unwrapped', gymnasium_env)
        if hasattr(underlying, 'get_contextual_diagnostics'):
            self.get_contextual_diagnostics = underlying.get_contextual_diagnostics
        elif hasattr(gymnasium_env, 'get_contextual_diagnostics'):
            self.get_contextual_diagnostics = gymnasium_env.get_contextual_diagnostics
        else:
            self.get_contextual_diagnostics = _no_op_diagnostics

        # Don't clobber an existing method on the unwrapped env
        if hasattr(gymnasium_env, 'unwrapped') and not hasattr(gymnasium_env.unwrapped, 'get_contextual_diagnostics'):
            gymnasium_env.unwrapped.get_contextual_diagnostics = self.get_contextual_diagnostics

    @staticmethod
    def _convert_space(space):
        import gymnasium
        if isinstance(space, gymnasium.spaces.Box):
            return gym_spaces.Box(
                low=space.low, high=space.high,
                shape=space.shape, dtype=space.dtype)
        elif isinstance(space, gymnasium.spaces.Dict):
            return gym_spaces.Dict({
                k: GymnasiumToGymWrapper._convert_space(v)
                for k, v in space.spaces.items()
            })
        elif isinstance(space, gymnasium.spaces.Discrete):
            return gym_spaces.Discrete(space.n)
        else:
            return space

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def reset(self, **kwargs):
        obs, _info = self._env.reset(**kwargs)
        return obs

    def set_next_reset_object_pose(self, xyz):
        """
        Forward forced object pose to the underlying gymnasium env.
        """
        target = getattr(self._env, 'unwrapped', self._env)
        if hasattr(target, 'set_next_reset_object_pose'):
            target.set_next_reset_object_pose(xyz)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def get_image(self, width=48, height=48):
        """Called by EnvRenderer._create_image() to render a frame."""
        img = self._env.unwrapped.mujoco_renderer.render('rgb_array')
        if img.shape[:2] != (height, width):
            from PIL import Image as PILImage
            img = np.array(PILImage.fromarray(img).resize(
                (width, height), PILImage.BILINEAR))
        return img  # HWC uint8

    def render(self, mode='human'):
        return self._env.render()

    def close(self):
        return self._env.close()

    def seed(self, seed=None):
        return []


class StretchEnvClass:
    """Shim so rlkit can call env_class(**env_kwargs). Module-level for pickling."""
    def __new__(cls, **kwargs):
        from scripts.stretch_env_factory import make_stretch_pick_env_with_images
        gymnasium_env = make_stretch_pick_env_with_images()
        return GymnasiumToGymWrapper(gymnasium_env)


flags.DEFINE_string('data_dir', './data', '')
flags.DEFINE_string('name', None, '')
flags.DEFINE_string('base_log_dir', None, '')
flags.DEFINE_bool('local', True, '')
flags.DEFINE_bool('gpu', True, '')
flags.DEFINE_bool('save_pretrained', True, '')
flags.DEFINE_bool('debug', False, '')
flags.DEFINE_bool('script', False, '')
flags.DEFINE_multi_string(
    'arg_binding', None, 'Variant binding to pass through.')

FLAGS = flags.FLAGS


def get_paths(data_dir):
    data_path = os.path.join(data_dir, 'stretch_align')
    demo_pkl = os.path.join(data_dir, 'stretch_align_trajectories5-1.pkl')
    demo_paths = [
        dict(path=demo_pkl,
             obs_dict=True,
             is_demo=True,
             use_latents=True)
    ]
    logging.info('Number of demonstration files: %d' % len(demo_paths))
    logging.info('demo_pkl: %s', demo_pkl)

    return data_path, demo_paths


def get_default_variant(demo_paths):
    default_variant = dict(
        imsize=48,
        env_kwargs=dict(
            test_env=True,
        ),
        policy_class=GaussianCNNPolicy,
        policy_kwargs=dict(
            hidden_sizes=[1024, 1024, 1024, 1024],
            std=0.15,
            max_log_std=-1,
            min_log_std=-13,
            std_architecture='shared',
            output_activation=None,
            layer_norm=True,
        ),
        qf_kwargs=dict(
            hidden_sizes=[1024, 1024, 1024, 1024],
            representation_dim=16,
            # repr_norm=False,
            repr_norm=True,
            repr_norm_temp=True,
            repr_log_scale=None,
            twin_q=True,
            layer_norm=True,
            img_encoder_arch='cnn',
            # init_w=1E-12,
            init_w=3E-3,
        ),
        network_type='contrastive_cnn',

        trainer_kwargs=dict(
            discount=0.99,
            lr=3E-4,
            gradient_clipping=None,
            soft_target_tau=5E-3,

            # Contrastive RL default hyperparameters
            bc_coef =1.0,
            # bc_coef=0.05,
            use_td=True,
            entropy_coefficient=0.0,
            target_entropy=0.0,

            augment_order=['crop'],
            augment_probability=0.5,
        ),

        # max_path_length=400, # original
        max_path_length=75,
        algo_kwargs=dict(
            batch_size=2048,
            # batch_size=2048, # original
            start_epoch=-300,
            num_epochs=0,  # just do pretraining
            # num_epochs=1,

            num_eval_steps_per_epoch=2000,
            # num_expl_steps_per_train_loop=2000,
            num_expl_steps_per_train_loop=0,
            num_trains_per_train_loop=1000,
            num_online_trains_per_train_loop=2000,
            # num_online_trains_per_train_loop=500,
            min_num_steps_before_training=4000,

            eval_epoch_freq=5,
            offline_expl_epoch_freq=5,
        ),
        replay_buffer_kwargs=dict(
            fraction_next_context=0.0,
            fraction_future_context=1.0,
            fraction_distribution_context=0.0,
            max_size=int(1E6),
            neg_from_the_same_traj=False,
        ),
        online_offline_split=True,
        reward_kwargs=dict(
            obs_type='image',
            reward_type='sparse',
            epsilon=6.0,  # image-space L2: same-traj median ~6.1, cross-traj min ~3.6
            terminate_episode=False,
        ),
        online_offline_split_replay_buffer_kwargs=dict(
            offline_replay_buffer_kwargs=dict(
                fraction_next_context=0.0,
                fraction_future_context=1.0,
                fraction_distribution_context=0.0,
                max_size=int(6E5),
                neg_from_the_same_traj=False,
            ),
            online_replay_buffer_kwargs=dict(
                fraction_next_context=0.0,
                fraction_future_context=1.0,
                fraction_distribution_context=0.0,
                max_size=int(4E5),
                neg_from_the_same_traj=False,
            ),
            sample_online_fraction=0.0
            # sample_online_fraction=0.6
        ),

        save_video=True,
        expl_save_video_kwargs=dict(
            save_video_period=25,
            pad_color=0,
        ),
        eval_save_video_kwargs=dict(
            save_video_period=25,
            pad_color=0,
        ),

        path_loader_kwargs=dict(
            delete_after_loading=False,
            # delete_after_loading=True,
            recompute_reward=True,
            demo_paths=demo_paths,
            split_max_steps=None,
            demo_train_split=0.95,
            add_demos_to_replay_buffer=True,  # set to false if we want to save paths.
            demos_saving_path=None,  # need to be a valid path if add_demos_to_replay_buffer is false
        ),

        renderer_kwargs=dict(
            create_image_format='HWC',
            output_image_format='CWH',
            flatten_image=True,
            width=48,
            height=48,
        ),

        add_env_demos=False,
        add_env_offpolicy_data=False,

        load_demos=True,

        evaluation_goal_sampling_mode='presampled_images',
        exploration_goal_sampling_mode='presampled_images',
        training_goal_sampling_mode='presample_latents',

        presampled_goal_kwargs=dict(
            eval_goals='',  # HERE
            eval_goals_kwargs={},
            expl_goals='',
            expl_goals_kwargs={},
            training_goals='',
            training_goals_kwargs={},
        ),

        launcher_config=dict(
            unpack_variant=True,
            region='us-west-1',
        ),
        logger_config=dict(
            snapshot_mode='gap',
            snapshot_gap=50,
            wandb=True,
            wandb_kwargs=dict(
                project='Stable_Contrastive_RL',
                name=os.environ.get('SLURM_JOB_ID', None),
            ),
        ),

        use_image=True,

        # Load up existing policy/q-network/value network vs train a new one
        pretrained_rl_path=None,

        eval_seeds=14,
        num_demos=9999,  # use all demos in the file

        # Video
        num_video_columns=5,
        save_paths=False,

        # Method Name
        method_name='stable_contrastive_rl',
    )

    return default_variant


def get_search_space():
    ########################################
    # Search Space
    ########################################
    search_space = {
        # Goals
        'ground_truth_expl_goals': [True],
    }

    return search_space


def process_variant(variant, data_path):
    # Error checking
    assert variant['algo_kwargs']['start_epoch'] % variant['algo_kwargs']['eval_epoch_freq'] == 0
    if variant['pretrained_rl_path'] is not None:
        assert variant['algo_kwargs']['start_epoch'] == 0
    if not variant['use_image']:
        assert variant['trainer_kwargs']['augment_probability'] == 0.0
    env_type = 'stretch'

    ########################################
    # Set the eval_goals.
    ########################################
    eval_goals = os.path.join(data_path, 'eval_goals5-1.pkl')
    print('eval_goals: ', eval_goals)

    ########################################
    # Goal sampling modes.
    ########################################
    variant['presampled_goal_kwargs']['eval_goals'] = eval_goals
    variant['path_loader_kwargs']['demo_paths'] = (
        variant['path_loader_kwargs']['demo_paths'][:variant['num_demos']])
    variant['online_offline_split_replay_buffer_kwargs']['offline_replay_buffer_kwargs']['max_size'] = int(float(
        variant['online_offline_split_replay_buffer_kwargs']['offline_replay_buffer_kwargs']['max_size']))
    variant['online_offline_split_replay_buffer_kwargs']['online_replay_buffer_kwargs']['max_size'] = int(float(
        variant['online_offline_split_replay_buffer_kwargs']['online_replay_buffer_kwargs']['max_size']))
    variant['replay_buffer_kwargs']['max_size'] = int(float(variant['replay_buffer_kwargs']['max_size']))

    if variant['ground_truth_expl_goals']:
        variant['exploration_goal_sampling_mode'] = 'presampled_images'
        variant['training_goal_sampling_mode'] = 'presampled_images'
        variant['presampled_goal_kwargs']['expl_goals'] = eval_goals
        variant['presampled_goal_kwargs']['training_goals'] = eval_goals

    # if variant['only_not_done_goals']:
    #     _old_mode = 'presampled_images'
    #     _new_mode = 'not_done_presampled_images'
    #
    #     if variant['training_goal_sampling_mode'] == _old_mode:
    #         variant['training_goal_sampling_mode'] = _new_mode
    #     if variant['exploration_goal_sampling_mode'] == _old_mode:
    #         variant['exploration_goal_sampling_mode'] = _new_mode
    #     if variant['evaluation_goal_sampling_mode'] == _old_mode:
    #         variant['evaluation_goal_sampling_mode'] = _new_mode

    ########################################
    # Environments.
    ########################################
    # GymnasiumToGymWrapper and StretchEnvClass are defined at module level
    # (top of file) so that torch.save() can pickle them.
    variant['env_class'] = StretchEnvClass
    variant['env_kwargs'] = {}

    ########################################
    # Image.
    ########################################
    if variant['use_image']:
        for demo_path in variant['path_loader_kwargs']['demo_paths']:
            demo_path['use_latents'] = False

    ########################################
    # Misc.
    ########################################
    if 'std' in variant['policy_kwargs']:
        if variant['policy_kwargs']['std'] <= 0:
            variant['policy_kwargs']['std'] = None


def main(_):
    data_path, demo_paths = get_paths(data_dir=FLAGS.data_dir)
    default_variant = get_default_variant(demo_paths)
    search_space = get_search_space()

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=default_variant,
    )

    logging.info('arg_binding: ')
    logging.info(FLAGS.arg_binding)

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variant = arg_util.update_bindings(variant,
                                           FLAGS.arg_binding,
                                           check_exist=True)
        process_variant(variant, data_path)
        variants.append(variant)

    run_variants(stable_contrastive_rl_experiment,
                 variants,
                 run_id=0,
                 process_args_fn=process_args)


if __name__ == '__main__':
    app.run(main)
