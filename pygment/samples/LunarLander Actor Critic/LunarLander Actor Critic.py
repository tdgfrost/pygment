import pygment as pm
import gymnasium as gym
import dill

train_model = True
env = gym.make('LunarLander-v2', max_episode_steps=500)
agent = pm.create_agent('actorcritic')
agent.load_env(env)
agent.add_network(nodes=[64, 64])
agent.compile('adam', learning_rate=0.001)

if train_model:
  agent.train(target_reward=200, episodes=100000, ep_update=64, gamma=0.99)
else:
  with open(
    '/opt/homebrew/Caskroom/miniforge/base/envs/rltextbook/lib/python3.9/site-packages/pygment/samples/LunarLander Actor Critic/agent_net.pkl',
    'rb') as f:
    agent.net = dill.load(f)

pm.animate(agent, 'LunarLander-v2', max_episode_steps=500)
