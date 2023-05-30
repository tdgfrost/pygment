import pygment as pm
import gymnasium as gym
import numpy as np

env = gym.make('CartPole-v1', max_episode_steps=200)
agent = pm.create_agent('PPO', 'cpu')
agent.load_env(env)
agent.load_model('/Users/thomasfrost/Documents/Github/pygment/pygment/pygment/samples/Discrete/CartPole PPO/2023_5_30_140033/model_best_3000.pt')

agent.compile('adam', learning_rate=0.001)

all_states, all_actions, all_rewards, all_next_states, all_cum_rewards = agent.explore(episodes=10000, parallel_envs=32)

for array, name in [[all_states, 'all_states.npy'], [all_actions, 'all_actions.npy'],
                    [all_rewards, 'all_rewards.npy'], [all_next_states, 'all_next_states.npy'],
                    [all_cum_rewards, 'all_cum_rewards.npy']]:
    np.save(f'./{name}', array)

