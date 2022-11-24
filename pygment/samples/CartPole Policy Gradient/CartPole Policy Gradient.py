import pygment as pm
import gymnasium as gym
from time import sleep
import torch

env = gym.make('CartPole-v1', max_episode_steps=10000)
agent = pm.create_agent('policy')
agent.load_env(env)
agent.add_network(nodes=[32, 32])
agent.compile('adam', learning_rate=0.01)
agent.train(target_reward=8000, episodes=100000, ep_update=64, gamma=0.9999)

pm.animate(agent, 'CartPole-v1', max_steps=10000)
