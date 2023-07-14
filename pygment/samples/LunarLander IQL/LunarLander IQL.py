import pygment as pm
import gymnasium as gym
import numpy as np
import os

#for template_reward in [20, 30, 50, 100, 130, 150, 200, 220]:
for template_reward in [150]:
    load_prior_model = False
    animate_only = False
    # template_reward = 100

    env = gym.make('LunarLander-v2')
    agent = pm.create_agent('iql', device='mps', path=os.path.abspath(os.path.join('../../../..', 'Informal experiments/mse_loss')))
    agent.load_env(env)

    agent.add_network(nodes=[64, 64])
    if load_prior_model:
        agent.load_model(criticpath1=None,
                         criticpath2=None,
                         valuepath=None,
                         actorpath=None,
                         behaviourpolicypath=None)

    agent.compile('adam', learning_rate=0.01, weight_decay=1e-8, clip=1)

    data_path = f'../GenerateStaticDataset/LunarLander/{template_reward} reward'

    loaded_data = {}
    """
    for key, filename in [['state', 'all_states.npy'], ['actions', 'all_actions.npy'],
                          ['rewards', 'all_rewards.npy'], ['next_state', 'all_next_states.npy'],
                          ['next_action', 'all_next_actions.npy'], ['dones', 'all_dones.npy'],
                          ['all_cum_rewards', 'all_cum_rewards.npy']]:
        loaded_data[key] = np.load(data_path + '/' + filename)
    """
    for key, filename in [['state', 'all_states.npy'], ['actions', 'all_actions.npy'],
                          ['original_rewards', 'all_rewards.npy'], ['next_state', 'all_next_states.npy'],
                          ['next_action', 'all_next_actions.npy'], ['dones', 'all_dones.npy'],
                          ['original_cum_rewards', 'all_cum_rewards.npy']]:
        loaded_data[key] = np.load(data_path + '/' + filename)

    # Scale to mean = 0, std = 1
    
    loaded_data['rewards'] = (loaded_data['original_rewards'] - loaded_data['original_rewards'].mean()) / loaded_data['original_rewards'].std()

    # Re-calculate the cum_rewards
    idxs = (np.where(loaded_data['dones'])[0]+1).tolist()
    idxs.insert(0, 0)
    idxs = np.array([(start_idx, end_idx) for start_idx, end_idx in zip(idxs[:-1], idxs[1:])])
    loaded_data['all_cum_rewards'] = loaded_data['original_cum_rewards'].copy()
    for start_idx, end_idx in idxs:
        cum_reward = 0
        for idx in range(end_idx-1, start_idx-1, -1):
            cum_reward = loaded_data['rewards'][idx] + 0.99 * cum_reward
            loaded_data['all_cum_rewards'][idx] = cum_reward

    # Find the correct scale
    # big_R_scale = loaded_data['all_cum_rewards'].max()

    data = [pm.Experience(state=loaded_data['state'][i],
                          action=loaded_data['actions'][i],
                          reward=loaded_data['rewards'][i],
                          next_state=loaded_data['next_state'][i],
                          next_action=loaded_data['next_action'][i],
                          cum_reward=loaded_data['all_cum_rewards'][i],
                          done=loaded_data['dones'][i]) for i in range(len(loaded_data['state']))]
                          #original_reward=loaded_data['original_rewards'][i],
                          #original_cumreward=loaded_data['original_cum_rewards'][i]) for i in range(len(loaded_data['state']))]

    #agent.clone_behaviour(data, batch_size=100000, epochs=1000000, evaluate=True, save=True)

    tau = 0.8
    desired_batch = 100000

    agent.train(data, critic=True, value=True, actor=True, evaluate=False, steps=1e6, batch_size=desired_batch,
                gamma=0.99, tau=tau, alpha=1, beta=1, update_iter=4, ppo_clip=1.2, ppo_clip_decay=1, save=True)

    _, _, _, _, rewards = agent.evaluate(episodes=800)
    for _ in range(10):
        pm.animate(agent, 'LunarLander-v2', max_episode_steps=500, directory=agent.path+f'/{template_reward}_video',
                   prefix=f'IQL_reward_{int(rewards.mean())}')
