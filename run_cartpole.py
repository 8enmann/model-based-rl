#!/usr/bin/env python
import logging, gym, click
import logger
from misc_util import set_global_seeds
from a2c import learn
from subproc_vec_env import SubprocVecEnv
from gym_wrappers import DownsampleWrapper, RenderWrapper, FrameStack, EncodedImaginationWrapper, DecodedImaginationWrapper, StaticWrapper
from policies import CnnPolicy, FcPolicy

@click.command()
@click.option('--env_id', default='CartPole-v0', help='openai/gym environment ID')
@click.option('--num_timesteps', default=8e6)
@click.option('--seed', default=0)
@click.option('--policy', default='fc')
@click.option('--nstack', default=2)
@click.option('--nsteps', default=5)
@click.option('--lrschedule', default='linear')
@click.option('--optimizer', default='adam')
@click.option('--num_cpu', default=8)
@click.option('--model_file', default='autoencoder.h5')
@click.option('--use_static_wrapper', is_flag=True)
@click.option('--use_encoded_imagination', is_flag=True)
@click.option('--use_decoded_imagination', is_flag=True)
def main(env_id, num_timesteps, seed, policy, nstack, nsteps, lrschedule, optimizer, num_cpu,
         model_file, use_static_wrapper, use_encoded_imagination, use_decoded_imagination):
    num_timesteps //= 4
    assert not (use_encoded_imagination and use_decoded_imagination)
    
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            if use_static_wrapper:
                env = StaticWrapper(env)
            if policy == 'cnn' or use_encoded_imagination:
                env = RenderWrapper(env, 400, 600)
                env = DownsampleWrapper(env, 4)
            if use_encoded_imagination or use_decoded_imagination:
                env = FrameStack(env, 3)
            if use_encoded_imagination:
                env = EncodedImaginationWrapper(env, model_file, num_cpu)
            if use_decoded_imagination:
                env = DecodedImaginationWrapper(env, model_file, num_cpu)
            gym.logger.setLevel(logging.WARN)
            return env
        return _thunk

    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    if policy == 'fc':
        policy_fn = FcPolicy
    if policy == 'cnn':
        policy_fn = CnnPolicy
    learn(policy_fn, env, seed, nsteps=nsteps, nstack=nstack, total_timesteps=num_timesteps, lrschedule=lrschedule, optimizer=optimizer, max_episode_length=195)
    env.close()

    
if __name__ == '__main__':
    main()
