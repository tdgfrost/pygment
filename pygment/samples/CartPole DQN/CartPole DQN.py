import pygment as pm
import gymnasium as gym

env = gym.make('CartPole-v1', max_episode_steps=300)
agent = pm.create_agent('doubleDQN')
agent.load_env(env)
agent.add_network([32, 32])
agent.compile('adam', learning_rate=0.001)
agent.train(200, tau=0.001, gamma=0.999999)

pm.animate(agent, 'CartPole-v1', max_episode_steps=300)