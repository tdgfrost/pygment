from tqdm import tqdm
import jax.numpy as jnp
import jax
import numpy as np
from gymnasium import envs

# Define config file - could change to FLAGS at some point
config = {'device': 'cpu',  # vs 'METAL'
          'seed': 42,
          'epochs': int(1e6),
          'batch_size': 512,
          'tau': 0.5,
          'expectile': 0.8,
          'temperature': 0.1,
          'gamma': 0.9999,
          'actor_lr': 3e-4,
          'value_lr': 3e-4,
          'critic_lr': 3e-4,
          'hidden_dims': (256, 256),
          }

# Set jax to CPU
jax.config.update('jax_platform_name', config['device'])

if __name__ == "__main__":
    from agent import IQLAgent
    from common import load_data, Batch

    # Create environment
    env = envs.make('LunarLander-v2', max_episode_steps=1000)

    # Load static dataset (dictionary) and convert to a 1D list of Experiences
    data = load_data(path='../samples/GenerateStaticDataset/LunarLander/140 reward',
                     scale='standardise',
                     gamma=config['gamma'])

    data = Batch(states=data['state'],
                 actions=data['actions'][:, np.newaxis],
                 rewards=data['rewards'],
                 discounted_rewards=data['discounted_rewards'],
                 next_states=data['next_state'],
                 next_actions=data['next_action'][:, np.newaxis],
                 dones=data['dones'])

    # Create agent
    agent = IQLAgent(observations=env.observation_space.sample(),
                     actions=env.action_space.sample()[np.newaxis],
                     action_dim=env.action_space.n,
                     dropout_rate=None,
                     opt_decay_schedule="cosine",
                     **config)

    # Train agent
    for epoch in tqdm(range(config['epochs'])):
        batch = agent.sample(data,
                             config['batch_size'])

        loss_info = agent.update(batch)

    """
    Other loss / metric recording aspects can go here
    """
