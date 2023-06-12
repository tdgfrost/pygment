import pygment as pm
import gymnasium as gym
import numpy as np

#for template_reward in [20, 30, 50, 100, 130, 150, 200, 220]:
for template_reward in [100]:
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

    agent.compile('adam', learning_rate=0.001, weight_decay=1e-8, clip=1)

    data_path = f'../GenerateStaticDataset/LunarLander/{template_reward} reward'

    loaded_data = {}
    for key, filename in [['state', 'all_states.npy'], ['actions', 'all_actions.npy'],
                          ['rewards', 'all_rewards.npy'], ['next_state', 'all_next_states.npy'],
                          ['next_action', 'all_next_actions.npy'], ['dones', 'all_dones.npy'],
                          ['all_cum_rewards', 'all_cum_rewards.npy']]:
        loaded_data[key] = np.load(data_path + '/' + filename)

    # Reduce scale of the rewards
    # loaded_data['rewards'] = loaded_data['rewards'] / 1000

    data = [pm.Experience(state=loaded_data['state'][i],
                          action=loaded_data['actions'][i],
                          reward=loaded_data['rewards'][i],
                          next_state=loaded_data['next_state'][i],
                          next_action=loaded_data['next_action'][i],
                          cum_reward=loaded_data['all_cum_rewards'][i],
                          done=loaded_data['dones'][i]) for i in range(len(loaded_data['state']))]

    #agent.clone_behaviour(data, batch_size=10240, epochs=1000000, evaluate=True, save=True)

    agent.train(data, critic=True, value=True, actor=True, evaluate=True, steps=1e6, batch_size=512,
                gamma=0.99, tau=0.9, alpha=1, beta=0.3, update_iter=10, ppo_clip=0.01, ppo_clip_decay=1, save=True)

    _, _, _, _, rewards = agent.evaluate(episodes=100)
    for _ in range(10):
        pm.animate(agent, 'LunarLander-v2', max_episode_steps=500, directory=agent.path+f'/{template_reward}_video',
                   prefix=f'IQL_reward_{int(rewards.mean())}')
