import pygment as pm
import gymnasium as gym
import torch
import os

train_new_model = False
animate_only = False

env = gym.make('LunarLander-v2', max_episode_steps=500)
agent = pm.create_agent('actorcritic')
agent.load_env(env)
if train_new_model:
  agent.add_network(nodes=[64, 64])
else:
  agent.load_model()

agent.compile('adam', learning_rate=0.01)

agent.train(target_reward=300, save_from=0, save_interval=10, episodes=100000, parallel_envs=100, gamma=0.99) if not animate_only else None

pm.animate(agent, 'LunarLander-v2', max_episode_steps=500)

#pm.animate_live(agent, 'CartPole-v1', max_episode_steps=5000)
