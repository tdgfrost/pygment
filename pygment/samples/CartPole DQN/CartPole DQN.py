import pygment as pm
import gymnasium as gym
from time import sleep

env = gym.make('CartPole-v1')
agent = pm.create_agent('doubleDQN')
agent.load_env(env)
agent.add_network([64])
agent.compile('adam', learning_rate=0.001)
agent.train(500, tau=0.05)

pm.animate(agent, 'CartPole-v1')