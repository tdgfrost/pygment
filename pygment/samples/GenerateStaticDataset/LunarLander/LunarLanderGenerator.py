import pygment as pm
import gymnasium as gym
import numpy as np
import os

for target_reward in [130]:
    env = gym.make('LunarLander-v2', max_episode_steps=500)
    agent = pm.create_agent('PPO', 'cpu')
    #target_reward = 100
    agent.load_env(env)
    agent.load_model(f'/Users/thomasfrost/Documents/Github/pygment/pygment/pygment/samples/LunarLander PPO/2023_6_01_120242/model_{target_reward}.pt')

    agent.compile('adam', learning_rate=0.001)

    if not os.path.isdir(f'./{target_reward} reward'):
        os.makedirs(f'./{target_reward} reward')

    all_states, all_actions, all_rewards, all_next_states, all_dones, all_cum_rewards = agent.explore(episodes=10000,
                                                                                                      parallel_envs=32)

    for array, name in [[all_states, 'all_states.npy'], [all_actions, 'all_actions.npy'],
                        [all_rewards, 'all_rewards.npy'], [all_next_states, 'all_next_states.npy'],
                        [all_dones, 'all_dones.npy'], [all_cum_rewards, 'all_cum_rewards.npy']]:
        np.save(f'./{target_reward} reward/{name}', array)

