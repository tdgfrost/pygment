import pygment as pm
import gymnasium as gym
import numpy as np

for target_reward in [180]:
    load_prior_model = True
    animate_only = True
    #target_reward = 220

    env = gym.make('LunarLander-v2', max_episode_steps=500)
    agent = pm.create_agent('ppo')
    agent.load_env(env)

    agent.add_network(nodes=[64, 64])
    if load_prior_model:
        agent.load_model(f'/Users/thomasfrost/Documents/Github/pygment/pygment/pygment/samples/LunarLander PPO/2023_6_13_162432/model_{target_reward}.pt')

    agent.compile('adam', learning_rate=0.01)

    agent.train(target_reward=target_reward, save_from=-100, save_interval=10, iterations=100000, sample_episodes=2000,
                parallel_envs=256, update_steps=1000, gamma=0.99) if not animate_only else None

    if not animate_only:
        _, _, rewards, _, _, dones, _ = agent.explore(episodes=2000, parallel_envs=256)

    for _ in range(10):
        pm.animate(agent, 'LunarLander-v2', max_episode_steps=500, directory=agent.path+f'/{target_reward}_video',
                   prefix=f'Baseline_reward_{target_reward}', target_reward=target_reward)

    # pm.animate_live(agent, 'CartPole-v1', max_episode_steps=5000)
