import numpy as np
import gymnasium as gym


'''
################
ENVIRONMENT
################
'''


class FrameStack(gym.Wrapper):
    def __init__(self, env, k=1):
        """ Repeats action k times"""
        super().__init__(env)
        self.k = k

    def reset(self):
        return self.env.reset()

    def step(self, action):
        rewards = []
        count = 0
        done = False
        prem_done = False
        while (not done) and (not prem_done) and (count != self.k):
            count += 1
            ob, reward, done, prem_done, info = self.env.step(action)
            rewards.append(reward)
        return ob, np.array(rewards).mean(), done, prem_done, info


class ClippedRewardsWrapper(gym.RewardWrapper):
    def reward(self, reward):
        """Change all the positive rewards to 1, negative to -1 and keep zero."""
        return np.sign(reward)


def wrap_env(env, stack_frames=1, reward_clipping=True):
    """Apply a common set of wrappers for Atari games."""
    env = FrameStack(env, stack_frames)
    if reward_clipping:
        env = ClippedRewardsWrapper(env)
    return env
