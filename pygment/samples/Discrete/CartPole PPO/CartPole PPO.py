import pygment as pm
import gymnasium as gym

train_new_model = False
animate_only = False

env = gym.make('CartPole-v1', max_episode_steps=3000)
agent = pm.create_agent('PPO', 'cpu')
agent.load_env(env)
if train_new_model:
  agent.add_network(nodes=[64, 64])
else:
  agent.load_model()

agent.compile('adam', learning_rate=0.001)

agent.train(target_reward=3000, save_from=100, save_interval=100, episodes=10000, parallel_envs=32,
            update_iter=10, update_steps=10000, batch_size=1024, gamma=0.99) if not animate_only else None

pm.animate(agent, 'CartPole-v1', max_episode_steps=4000)

#pm.animate_live(agent, 'CartPole-v1', max_episode_steps=4000)

