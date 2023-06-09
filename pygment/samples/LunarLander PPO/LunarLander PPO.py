import pygment as pm
import gymnasium as gym
import torch
import os

load_prior_model = False
animate_only = True

env = gym.make('LunarLander-v2', max_episode_steps=500)
agent = pm.create_agent('ppo')
agent.load_env(env)

agent.add_network(nodes=[64, 64])
if load_prior_model:
    agent.load_model('/Users/thomasfrost/Documents/Github/pygment/pygment/pygment/samples/LunarLander PPO/2023_6_01_120242/model_150.pt')

agent.compile('adam', learning_rate=0.01)

agent.train(target_reward=220, save_from=0, save_interval=10, episodes=100000, parallel_envs=100,
            gamma=0.99) if not animate_only else None

for _ in range(10):
    pm.animate(agent, 'LunarLander-v2', max_episode_steps=500, directory=agent.path+f'/100_video',
               prefix='Baseline_reward_100')

# pm.animate_live(agent, 'CartPole-v1', max_episode_steps=5000)
