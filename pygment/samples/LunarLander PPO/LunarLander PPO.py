import pygment as pm
import gymnasium as gym
import numpy as np

for target_reward in [-100]:
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

    _, _, rewards, _, _, dones, _ = agent.explore(episodes=2000, parallel_envs=256)

    # Temporary code to scale cum_rewards - involves re-calculating cum_rewards
    idxs = np.where(dones)[0].tolist()
    idxs.insert(0, -1)
    idxs = [(idxs[i] + 1, idxs[i + 1]) for i in range(len(idxs) - 1)]
    total_rewards = []
    for idx_tuple in idxs:
        total_reward = np.array(rewards[idx_tuple[0]:idx_tuple[1] + 1]).sum()
        total_rewards.append(total_reward)
    total_rewards = np.array(total_rewards)
    target_reward = int(total_rewards.mean())

    for _ in range(10):
        pm.animate(agent, 'LunarLander-v2', max_episode_steps=500, directory=agent.path+f'/{target_reward}_video',
                   prefix=f'Baseline_reward_{target_reward}')

    # pm.animate_live(agent, 'CartPole-v1', max_episode_steps=5000)
