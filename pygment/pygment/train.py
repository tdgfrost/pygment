from agent import IQLAgent
from common import load_data, Experience

from tqdm import tqdm
import jax.numpy as jnp
import jax
import numpy as np
from gymnasium import envs

# Define config file - could change to FLAGS at some point
config = {'device': 'cpu',  # vs 'METAL'
          'epochs': int(1e6),
          'batch_size': 256,
          'tau': 0.5,
          'expectile': 0.8,
          'temperature': 0.1,
          'gamma': 0.9999,
          'actor_lr': 3e-4,
          'value_lr': 3e-4,
          'critic_lr': 3e-4,
          }

# Set jax to CPU
jax.config.update('jax_platform_name', config['device'])

# Create environment
env = envs.make('LunarLander-v2', max_episode_steps=1000)

# Load static dataset (dictionary) and convert to a 1D list of Experiences
data = load_data(path='',
                 scale='standardise',
                 gamma=config['gamma'])

data = [Experience(state=data['state'][i],
                   action=data['actions'][i],
                   reward=data['discounted_rewards'][i],
                   next_state=data['next_state'][i],
                   next_action=data['next_action'][i],
                   done=data['dones'][i],
                   original_reward=data['original_rewards'][i],
                   original_discounted_reward=data['discounted_rewards'][i],
                   ) for i in range(len(data['dones']))]

# Create agent
agent = IQLAgent(seed=42,
                 observations=env.observation_space.sample()[np.newaxis],
                 actions=env.action_space.sample()[np.newaxis],
                 actor_lr=config['actor_lr'],
                 value_lr=config['value_lr'],
                 critic_lr=config['critic_lr'],
                 hidden_dims=(256, 256),
                 discount=config['gamma'],
                 tau=config['tau'],
                 expectile=config['expectile'],
                 dropout_rate=None,
                 max_steps=None,
                 opt_decay_schedule="cosine",
                 )

# Train agent
for epoch in tqdm(range(config['epochs'])):
    batch = agent.sample(data,
                         config['batch_size'])

    loss_info = agent.update(batch,
                             **config)

"""
Other loss / metric recording aspects can go here
"""
