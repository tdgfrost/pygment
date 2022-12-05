import pygment as pm
import gymnasium as gym
import torch
import os

train_new_model = True
animate_only = False

env = gym.make('LunarLanderContinuous-v2', max_episode_steps=500)
agent = pm.create_agent('actorcriticcontinuous', 'cpu')
agent.load_env(env)
if train_new_model:
  agent.add_network(nodes=[512, 256, 64])
else:
  agent.load_model()

agent.compile('adam', learning_rate=0.01)

agent.train(target_reward=300, save_from=50, save_interval=50, episodes=100000, parallel_envs=100, gamma=0.99) if not animate_only else None

pm.animate(agent, 'LunarLanderContinuous-v2', max_episode_steps=500)

#pm.animate_live(agent, 'LunarLanderContinuous-v2', max_episode_steps=500)
