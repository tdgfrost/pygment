from typing import Dict, Any, List

import gymnasium
import numpy as np
import jax.numpy as jnp
import jax
from common import Batch
from stable_baselines3.common.env_util import make_vec_env


def generate_episodes(policy, envs, key=None, gamma=0.99):
    """
    Evaluate the agent across vectorised episodes.

    :param policy: policy to evaluate.
    :param envs: vectorised environments.
    :param key: random key.
    :return: array of total rewards for each episode.
    """

    # Initial parameters
    key = jax.random.PRNGKey(123) if key is None else key
    num_envs = envs.num_envs

    states = envs.reset()
    all_states = [[] for _ in range(num_envs)]
    all_actions = [[] for _ in range(num_envs)]
    all_action_logprobs = [[] for _ in range(num_envs)]
    all_rewards = [[] for _ in range(num_envs)]
    flattened_rewards = [[] for _ in range(num_envs)]
    all_next_states = [[] for _ in range(num_envs)]
    all_dones = [[] for _ in range(num_envs)]
    active_idx = np.array([i for i in range(num_envs)])
    longest_episode = 0

    # Complete all episodes
    while active_idx.any():
        longest_episode += 1
        # Step through environments
        actions, action_logprobs = policy.sample_action(states, key)

        next_states, _, dones, info = envs.step(actions)
        rewards = [d['sequential_reward'] for d in info]

        # Update finished environments
        prem_dones = np.array([d['TimeLimit.truncated'] for d in info])
        dones = np.any((dones, prem_dones), axis=0)

        # Update record
        [all_states[i].append(states[i]) for i in active_idx]
        [all_actions[i].append(actions[i]) for i in active_idx]
        [all_action_logprobs[i].append(np.array(action_logprobs)[i][actions[i]]) for i in active_idx]
        [all_rewards[i].append(rewards[i]) for i in active_idx]
        [flattened_rewards[i].extend(rewards[i]) for i in active_idx]
        [all_next_states[i].append(next_states[i]) for i in active_idx]
        [all_dones[i].append(dones[i]) for i in active_idx]

        # Update remaining parameters
        active_idx = np.where(~np.array([done[-1] for done in all_dones]))[0]
        states = next_states
        key = jax.random.split(key, num=1)[0]

    # For variable time delays, find the index of when an action is taken (relative to each episode step)
    flattened_idx = [[i for i in range(len(flattened_ep)) if flattened_ep[i] == seq_ep[i][0]]
                     for flattened_ep, seq_ep in zip(flattened_rewards, all_rewards)]

    # Zero-pad the rewards to the length of the longest episode and convert to Jax array
    flattened_rewards = jnp.array([[ep[i] if i < len(ep) else 0 for i in range(longest_episode)]
                                   for ep in flattened_rewards])

    # Calculate the total reward for each episode
    episode_rewards = np.sum(flattened_rewards, axis=1).reshape(-1, 1)
    """
    # Using the zero-padded rewards, calculate the discounted rewards
    discounted_rewards = jax.lax.scan(lambda agg, reward: (agg * gamma + reward, agg * gamma + reward),
                                      jnp.zeros(shape=num_envs), flattened_rewards.transpose(),
                                      reverse=True)[1].transpose()

    # Then convert the discounted rewards back to a non-padded list of lists
    discounted_rewards = [[ep[r_idx].item() for r_idx in flattened_idx[ep_idx]]
                          for ep_idx, ep in enumerate(np.array(discounted_rewards))]
    """
    # Calculate the value function for each state in each episode
    current_v = [np.array(policy.value(ep)[1]) for ep in all_states]

    # Calculate the future discounted value for each step of each episode
    next_v = [np.array(policy.value(ep)[1]) * ~np.array(dones) for ep, dones in zip(all_next_states, all_dones)]

    # Calculate the current discounted value for each step of each episode
    discounted_rewards = [[r + gamma * future_r for r, future_r in zip(ep_r, ep_next_v)]
                          for ep_r, ep_next_v in zip(np.array(flattened_rewards), next_v)]

    # Calculate the advantage value for each step in each episode
    advantages = [[disc_r - value for disc_r, value in zip(ep_r, ep_val)]
                  for ep_r, ep_val in zip(discounted_rewards, current_v)]

    # Normalise the advantage values
    advantages_flattened = np.array([adv for ep in advantages for adv in ep])
    advantages = [[(adv - advantages_flattened.mean()) / (advantages_flattened.std() + 1e-8)
                   for adv in ep] for ep in advantages]

    # Return a Batch
    return Batch(states=all_states,
                 actions=all_actions,
                 action_logprobs=all_action_logprobs,
                 rewards=all_rewards,
                 discounted_rewards=discounted_rewards,
                 next_states=all_next_states,
                 dones=all_dones,
                 episode_rewards=episode_rewards,
                 advantages=advantages), key


class EpisodeGenerator:
    def __init__(self, envs=make_vec_env('LunarLander-v2', n_envs=10), gamma=0.99):
        self.envs = envs
        self.gamma = gamma

    def __call__(self, policy, key):
        return generate_episodes(policy,
                                 self.envs,
                                 key=key,
                                 gamma=self.gamma)


class VariableTimeSteps(gymnasium.Wrapper):
    """
    Custom wrapper to implement a variable time-step environment.
    """

    def __init__(self, env, fn=None, max_time_steps=10):
        """
        :param env: Gymnasium environment
        :param fn: Custom function to generate a non-random time-delay - it is assumed that this function already
        includes a maximum time-delay.
        :param max_time_steps: If no custom function provided, a random time-delay is generated between 0
        and max_time_steps.
        """
        super().__init__(env)
        self.fn = fn
        self.max_time_steps = max_time_steps

    def step(self, action):
        state: np.ndarray
        reward: float
        done: bool
        prem_done: bool
        info: Dict[str, Any]
        sequential_reward: List[float] = []

        state, reward, done, prem_done, info = super().step(action)
        sequential_reward += [reward]
        total_reward = reward

        if done or prem_done:
            info['time_steps'] = 1
            info['sequential_reward'] = sequential_reward
            return state, total_reward, done, prem_done, info

        time_steps = self._generate_time_steps(state)
        """TEMPORARY"""
        time_steps = 0
        """TEMPORARY"""

        for step in range(2, time_steps + 2):
            state, reward, done, prem_done, info = super().step(action)
            sequential_reward += [reward]
            total_reward += reward

            if done or prem_done:
                info['time_steps'] = step
                info['sequential_reward'] = sequential_reward
                return state, total_reward, done, prem_done, info

        info['time_steps'] = time_steps + 1
        info['sequential_reward'] = sequential_reward
        return state, total_reward, done, prem_done, info

    def _generate_time_steps(self, state):
        if self.fn is None:
            return np.random.randint(0, self.max_time_steps + 1)

        else:
            return self.fn(state)

