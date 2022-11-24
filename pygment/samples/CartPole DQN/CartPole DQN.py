import pygment as pm
import gymnasium as gym
from time import sleep

env = gym.make('CartPole-v1', max_episode_steps=700)
agent = pm.create_agent('doubleDQN')
agent.load_env(env)
agent.add_network([64])
agent.compile('adam', learning_rate=0.01)
agent.train(500, tau=200)

pm.animate(agent, 'CartPole-v1', max_episode_steps=700)