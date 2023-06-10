import pygment as pm
import gymnasium as gym

load_prior_model = False
animate_only = False

env = gym.make('CartPole-v1', max_episode_steps=3000)
agent = pm.create_agent('PPO', 'cpu')
agent.load_env(env)

agent.add_network(nodes=[64, 64])
if load_prior_model:
    agent.load_model()

agent.compile('adam', learning_rate=0.001)

agent.train(target_reward=500, save_from=10, save_interval=50, episodes=10000, parallel_envs=32,
            update_iter=10, update_steps=10000, batch_size=1024, gamma=0.99) if not animate_only else None

pm.animate(agent, 'CartPole-v1', max_episode_steps=4000, prefix='')

# pm.animate_live(agent, 'CartPole-v1', max_episode_steps=4000)
