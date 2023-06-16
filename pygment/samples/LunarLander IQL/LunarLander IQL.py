import pygment as pm
import gymnasium as gym
import numpy as np

#for template_reward in [20, 30, 50, 100, 130, 150, 200, 220]:
for template_reward in [50]:
    load_prior_model = True
    animate_only = False
    # template_reward = 100

    env = gym.make('LunarLander-v2')
    agent = pm.create_agent('iql', device='mps')
    agent.load_env(env)

    agent.add_network(nodes=[64, 64])
    if load_prior_model:
        agent.load_model(criticpath1=None,
                         criticpath2=None,
                         valuepath=None,
                         actorpath=None,
                         behaviourpolicypath='./BehaviourPolicy/behaviour_policy_0.88427.pt')

    agent.compile('adam', learning_rate=0.01, weight_decay=1e-8, clip=1)

    data_path = f'../GenerateStaticDataset/LunarLander/{template_reward} reward'

    loaded_data = {}
    for key, filename in [['state', 'all_states.npy'], ['actions', 'all_actions.npy'],
                          ['rewards', 'all_rewards.npy'], ['next_state', 'all_next_states.npy'],
                          ['next_action', 'all_next_actions.npy'], ['dones', 'all_dones.npy'],
                          ['all_cum_rewards', 'all_cum_rewards.npy']]:
        loaded_data[key] = np.load(data_path + '/' + filename)

    # Reduce scale of the rewards
    loaded_data['rewards'] = (loaded_data['rewards'] - loaded_data['rewards'].mean()) / loaded_data['rewards'].std()
    # Temporary code to scale cum_rewards - involves re-calculating cum_rewards
    idxs = np.where(loaded_data['dones'])[0].tolist()
    idxs.insert(0, -1)
    idxs = [(idxs[i] + 1, idxs[i + 1]) for i in range(len(idxs) - 1)]
    for idx_tuple in idxs:
        cum_reward = 0
        for idx in range(idx_tuple[1], idx_tuple[0]-1, -1):
            cum_reward = loaded_data['rewards'][idx] + 0.99 * cum_reward
            loaded_data['all_cum_rewards'][idx] = cum_reward

    data = [pm.Experience(state=loaded_data['state'][i],
                          action=loaded_data['actions'][i],
                          reward=loaded_data['rewards'][i],
                          next_state=loaded_data['next_state'][i],
                          next_action=loaded_data['next_action'][i],
                          cum_reward=loaded_data['all_cum_rewards'][i],
                          done=loaded_data['dones'][i]) for i in range(len(loaded_data['state']))]

    #agent.clone_behaviour(data, batch_size=10240, epochs=1000000, evaluate=True, save=True)

    agent.train(data, critic=True, value=True, actor=True, evaluate=True, steps=1e6, batch_size=64,
                gamma=0.99, tau=0.9, alpha=1, beta=5, update_iter=4, ppo_clip=1.2, ppo_clip_decay=1, save=True)

    _, _, _, _, rewards = agent.evaluate(episodes=800)
    for _ in range(10):
        pm.animate(agent, 'LunarLander-v2', max_episode_steps=500, directory=agent.path+f'/{template_reward}_video',
                   prefix=f'IQL_reward_{int(rewards.mean())}')
