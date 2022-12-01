import pygment as pm
import gymnasium as gym
import torch

train_new_model = True
animate_only = False

env = gym.make('LunarLander-v2', max_episode_steps=500)
agent = pm.create_agent('actorcritic')
agent.load_env(env)
agent.add_network(nodes=[64, 64]) if train_new_model else agent.net.load_state_dict(torch.load('./model.pt'))
agent.compile('adam', learning_rate=0.01)

agent.train(target_reward=200, episodes=100000, parallel_envs=100, gamma=0.99) if not animate_only else None

torch.save(agent.net.state_dict(), './model.pt') if not animate_only else None

pm.animate(agent, 'LunarLander-v2', max_episode_steps=500)

#pm.animate_live(agent, 'CartPole-v1', max_episode_steps=5000)
