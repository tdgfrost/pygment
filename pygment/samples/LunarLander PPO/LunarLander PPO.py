import pygment as pm
import gymnasium as gym
import torch
import os

train_new_model = True
animate_only = False

env = gym.make('LunarLander-v2', max_episode_steps=500)
agent = pm.create_agent('PPO', 'cpu')
agent.load_env(env)
if train_new_model:
  agent.add_network(nodes=[64, 64])
else:
  agent.load_model()

agent.compile('adam', learning_rate=0.001)

agent.train(target_reward=280, save_from=100, save_interval=50, episodes=10000, parallel_envs=100,
            update_iter=10, update_steps=10000, batch_size=1024, gamma=0.99) if not animate_only else None

pm.animate(agent, 'LunarLander-v2', max_episode_steps=500)

#pm.animate_live(agent, 'LunarLander-v2', max_episode_steps=5000)
