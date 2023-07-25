import pygment as pm
import gymnasium as gym
import numpy as np

for target_reward in [140]:
    load_prior_model = True
    animate_only = True
    #target_reward = 220

    env = gym.make('LunarLander-v2', max_episode_steps=1000)
    agent = pm.create_agent('ppo')
    agent.load_env(env)

    agent.add_network(nodes=[64, 64])
    if load_prior_model:
        agent.load_model(f'/Users/thomasfrost/Documents/Github/pygment/pygment/pygment/samples/LunarLander PPO/2023_6_12_164457/model_{target_reward}.pt')

    agent.compile('adam', learning_rate=0.01)

    agent.train(target_reward=target_reward, save_from=-100, save_interval=10, iterations=100000, sample_episodes=2000,
                parallel_envs=256, update_steps=1000, gamma=0.99) if not animate_only else None

    #_, _, _, _, _, dones, total_rewards = agent.explore(episodes=1000, parallel_envs=512, gamma=1)
    #print((total_rewards[0] + total_rewards[np.where(dones[:-1])[0]+1].sum()) / dones.sum())

    for _ in range(10):
        pm.animate(agent, 'LunarLander-v2', max_episode_steps=1000, directory=agent.path+f'/{target_reward}_video',
                   prefix=f'Baseline_reward_{target_reward}', target_reward=target_reward)

    # pm.animate_live(agent, 'CartPole-v1', max_episode_steps=5000)
