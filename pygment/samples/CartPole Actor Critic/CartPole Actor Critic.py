import pygment as pm
import gymnasium as gym
import dill

env = gym.make('CartPole-v1', max_episode_steps=10000)
agent = pm.create_agent('actorcritic')
agent.load_env(env)
agent.add_network(nodes=[32, 32])
agent.compile('adam', learning_rate=0.01)

agent.train(target_reward=8000, episodes=100000, parallel_envs=50, gamma=1)

pm.animate(agent, 'CartPole-v1', max_episode_steps=10000)



#with open(
  #'/opt/homebrew/Caskroom/miniforge/base/envs/rltextbook/lib/python3.9/site-packages/pygment/samples/CartPole Actor Critic/agent_net.pkl',
  #'rb') as f:
  #agent.net = dill.load(f)
