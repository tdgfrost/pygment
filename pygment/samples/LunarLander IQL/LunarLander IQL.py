import pygment as pm
import gymnasium as gym
import numpy as np

#for template_reward in [20, 30, 50, 100, 130, 150, 200, 220]:
for template_reward in [130]:
    train_new_model = False
    animate_only = False
    # template_reward = 100

    env = gym.make('LunarLander-v2')
    agent = pm.create_agent('iql', device='mps')
    agent.load_env(env)

    agent.add_network(nodes=[64, 64])
    if not train_new_model:
        agent.load_model(actorpath1='./2023_6_05_164504/actor1_target_4489.34731.pt',
                         actorpath2='./2023_6_05_164504/actor2_target_4489.34731.pt',
                         criticpath='./2023_6_05_164504/critic_4489.34731.pt',
                         policypath='./2023_6_05_164504/policy_loss_policy_0.84957.pt')

    agent.compile('adam', learning_rate=0.01, weight_decay=1e-8, clip=1)

    data_path = f'../GenerateStaticDataset/LunarLander/{template_reward} reward'

    loaded_data = {}
    for key, filename in [['state', 'all_states.npy'], ['actions', 'all_actions.npy'],
                          ['rewards', 'all_rewards.npy'], ['next_state', 'all_next_states.npy'],
                          ['dones', 'all_dones.npy']]:
        loaded_data[key] = np.load(data_path + '/' + filename)

    data = [pm.Experience(state=loaded_data['state'][i],
                          action=loaded_data['actions'][i],
                          reward=loaded_data['rewards'][i],
                          next_state=loaded_data['next_state'][i],
                          done=loaded_data['dones'][i]) for i in range(len(loaded_data['state']))]

    #agent.train_qv(data, epochs=1000, batch_size=512, gamma=0.99, tau=0.9, alpha=0.01, save=True)
    #agent.train_policy(data, epochs=1000, batch_size=512, beta=0.001, save=True)
    _, _, _, _, rewards = agent.evaluate(episodes=100)

    for _ in range(10):
        pm.animate(agent, 'LunarLander-v2', max_episode_steps=500, directory=agent.path+f'/{template_reward}_video',
                   prefix=f'IQL_reward_{int(rewards.mean())}')
