import numpy as np
from collections import deque
from PIL import Image
import gym
from gym import spaces
from skimage.transform import resize
from keras.models import Model, load_model
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def get_session(num_threads, gpu_fraction=0.08):
    """Force tensorflow not to take up the whole GPU on every thread.

    https://groups.google.com/forum/#!topic/keras-users/MFUEY9P1sc8
    """
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))


class DecodedImaginationWrapper(gym.ObservationWrapper):
    """Load pre-trained environment model and use it to predict next frames."""

    def __init__(self, env, model_file, num_threads):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        KTF.set_session(get_session(num_threads))
        self.model = load_model(model_file)

        # take all actions as a batch, skip model batch dim
        shape = tuple(self.model.output.shape.as_list()[2:-1]) + (self.action_space.n,)
        self.observation_space = spaces.Box(low=-3e38, high=3e38, shape=shape)

    def _observation(self, obs):
        # Put framestack at the end
        obs = np.moveaxis(obs, -1, 0)
        # Preprocess
        obs = 1 - obs[..., :96, :144, :] / 255.
        return np.squeeze(
            np.concatenate(
                self.model.predict([
                    np.stack((obs,) * self.action_space.n),
                    np.identity(self.action_space.n)]
                ), axis=-1
            ), axis=0)


class EncodedImaginationWrapper(gym.ObservationWrapper):
    """Load pre-trained environment model and use it to encode each observation."""

    def __init__(self, env, model_file, num_threads):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        KTF.set_session(get_session())
        self.model = load_model(model_file)

        self.encoder_model = Model(self.model.input, self.model.get_layer('bottleneck').output, name='encoder')

        # take all actions as a batch, skip model batch dim
        shape = (self.action_space.n * self.encoder_model.output.shape.as_list()[1],)
        self.observation_space = spaces.Box(low=-3e38, high=3e38, shape=shape)

    def _observation(self, obs):
        # Put framestack at the end
        obs = np.moveaxis(obs, -1, 0)
        # Preprocess
        obs = 1 - obs[..., :96, :144, :] / 255.
        return np.concatenate(self.encoder_model.predict(
            [np.stack((obs,) * self.action_space.n),
             np.identity(self.action_space.n)]))

class StaticWrapper(gym.ObservationWrapper):
        """Take only position and angle observations."""

        def __init__(self, env):
            """Buffer observations and stack across channels (last axis)."""
            gym.Wrapper.__init__(self, env)
            self.observation_space = spaces.Box(low=-3e38, high=3e38, shape=(2,))

        def _observation(self, obs):
            """Take 1st and 3rd channels of the observation.

            Channels are position, velocity, angle, and angular velocity.
            https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L70
            """
            return obs[::2]


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def _reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip       = skip

    def _step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def _reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class ClipRewardEnv(gym.RewardWrapper):
    def _reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.res = 84
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.res, self.res, 1))

    def _observation(self, obs):
        frame = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
        frame = np.array(Image.fromarray(frame).resize((self.res, self.res),
            resample=Image.BILINEAR), dtype=np.uint8)
        return frame.reshape((self.res, self.res, 1))

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        assert shp[2] == 1  # can only stack 1-channel frames
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], k))

    def _reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        for _ in range(self.k): self.frames.append(ob)
        return self._observation()

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._observation(), reward, done, info

    def _observation(self):
        assert len(self.frames) == self.k
        return np.moveaxis(np.stack(self.frames), 0, -1)

def wrap_deepmind(env, episode_life=True, clip_rewards=True):
    """Configure environment for DeepMind-style Atari.

    Note: this does not include frame stacking!"""
    assert 'NoFrameskip' in env.spec.id  # required for DeepMind-style skip
    if episode_life:
        env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    return env


class RenderWrapper(gym.ObservationWrapper):
    def __init__(self, env, w, h):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(w, h, 3))

    def _observation(self, obs):
        return self.env.render(mode='rgb_array')


class DownsampleWrapper(gym.ObservationWrapper):
    """Resize image, grayscale"""

    def __init__(self, env, scale):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.scale = scale
        old_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(
            old_shape[0] // scale, old_shape[1] // scale, 1))

    def _observation(self, obs):
        return np.uint8(resize(
            np.mean(obs, axis=-1, keepdims=True),
            (obs.shape[0] // self.scale, obs.shape[1] // self.scale),
            mode='edge'))
