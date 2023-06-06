import pygment as pm
import gymnasium as gym
import numpy as np

for template_reward in [100]:
    load_prior_model = False
    animate_only = False
    # template_reward = 100

    env = gym.make('CartPole-v1')
    agent = pm.create_agent('iql', device='mps')
    agent.load_env(env)

    agent.add_network(nodes=[64, 64])
    if load_prior_model:
        agent.load_model(criticpath1='./2023_6_06_130051/critic1_389.pt',
                         criticpath2='./2023_6_06_130051/critic2_389.pt',
                         valuepath='./2023_6_06_130051/value_389.pt',
                         actorpath='./2023_6_06_130051/actor_389.pt')

    agent.compile('adam', learning_rate=0.01)

    data_path = f'../GenerateStaticDataset/CartPole/{template_reward} reward'

    loaded_data = {}
    for key, filename in [['state', 'all_states.npy'], ['actions', 'all_actions.npy'],
                          ['rewards', 'all_rewards.npy'], ['next_state', 'all_next_states.npy'],
                          ['dones', 'all_dones.npy']]:
        loaded_data[key] = np.load(data_path + '/' + filename)

    # Reduce scale of the rewards
    loaded_data['rewards'] = loaded_data['rewards'] / 1000

    data = [pm.Experience(state=loaded_data['state'][i],
                          action=loaded_data['actions'][i],
                          reward=loaded_data['rewards'][i],
                          next_state=loaded_data['next_state'][i],
                          done=loaded_data['dones'][i]) for i in range(len(loaded_data['state']))]

    agent.train(data, critic=True, value=True, actor=True, evaluate=True, steps=1e6, batch_size=64,
                gamma=0.99, tau=0.8, alpha=0.005, beta=0.3, save=True)

    _, _, _, _, rewards = agent.evaluate(episodes=100)

    pm.animate(agent, 'CartPole-v1', max_episode_steps=3000, directory=agent.path + f'/{template_reward}_video',
               prefix=f'IQL_reward_{int(rewards.mean())}')
