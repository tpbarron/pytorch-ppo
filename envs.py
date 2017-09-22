import os

import gym
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import *


def make_env(env_id, seed, log_dir):
    env = gym.make(env_id)
    env.seed(seed)
    env = bench.Monitor(env, os.path.join(log_dir, "monitor.json"))
    env = wrap_deepmind(env)
    env = WrapPyTorch(env)
    return env

class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 84, 84])

    def _observation(self, observation):
        return observation.transpose(2, 0, 1)
