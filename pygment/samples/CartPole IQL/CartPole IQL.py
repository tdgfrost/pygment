import pygment as pm
import gymnasium as gym
import numpy as np

train_new_model = True
animate_only = False
template_reward = 100

env = gym.make('CartPole-v1')
agent = pm.create_agent('iql', device='mps')
agent.load_env(env)

if train_new_model:
    agent.add_network(nodes=[64, 64])
else:
    agent.load_model()

agent.compile('adam', learning_rate=0.001)

data_path = f'../GenerateStaticDataset/CartPole/{template_reward} reward'

loaded_data = {}
for key, filename in [['state', 'all_states.npy'], ['actions', 'all_actions.npy'],
                      ['rewards', 'all_rewards.npy'], ['cum_rewards', 'all_cum_rewards.npy'],
                      ['next_state', 'all_next_states.npy']]:
    loaded_data[key] = np.load(data_path + '/' + filename)

data = [pm.Experience(state=loaded_data['state'][i],
                      action=loaded_data['actions'][i],
                      reward=loaded_data['rewards'][i],
                      cum_reward=loaded_data['cum_rewards'][i],
                      next_state=loaded_data['next_state'][i],
                      done=None) for i in range(len(loaded_data['state']))]

agent.train(data, epochs=100, batch_size=10240, gamma=0.99, tau=0.99, beta=100, save=True)
agent.evaluate(episodes=100)

pm.animate(agent, 'CartPole-v1', max_episode_steps=3000)
