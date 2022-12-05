import pygment as pm
import gymnasium as gym
import torch
import os

train_new_model = True
animate_only = False

env = gym.make('BipedalWalker-v3', max_episode_steps=1600)
agent = pm.create_agent('actorcriticcontinuous', 'cpu')
agent.load_env(env)
if train_new_model:
  agent.add_network(nodes=[1024, 1024, 1024])
else:
  agent.load_model()

agent.compile('adam', learning_rate=0.01)

agent.train(target_reward=300, save_from=50, save_interval=50, episodes=100000, parallel_envs=5, gamma=0.99) if not animate_only else None

pm.animate(agent, 'BipedalWalker-v3', max_episode_steps=1600)

#pm.animate_live(agent, 'BipedalWalker-v2', max_episode_steps=5000)
