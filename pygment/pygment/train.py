from tqdm import tqdm
import numpy as np
from gymnasium import envs
from tensorboardX import SummaryWriter
import os
import jax
import jax.numpy as jnp
import gymnasium

# Set jax to CPU
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_debug_nans", True)
# jax.config.update('jax_disable_jit', True)

# Define config file - could change to FLAGS at some point
config = {'seed': 123,
          'epochs': int(1e6),
          'batch_size': int(1e5),
          'expectile': 0.8,
          'gamma': 0.9999,
          'actor_lr': 5e-3,
          'value_lr': 5e-3,
          'critic_lr': 5e-3,
          'hidden_dims': (256, 256),
          'clipping': 1,
          }

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

    # Prepare logging tensorboard
    summary_writer = SummaryWriter('../experiments/tensorboard/current',
                                   write_to_disk=True)
    os.makedirs('../experiments/tensorboard/current/', exist_ok=True)

    # Train agent
    for value, critic, actor in [[True, False, False], [False, True, False], [False, False, True]]:

        loss_key = f"{'value' if value else ('critic' if critic else 'actor')}_loss"
        best_loss = jnp.inf
        count = 0
        for epoch in tqdm(range(config['epochs'])):
            batch = agent.sample(data,
                                 config['batch_size'])

            loss_info = agent.update_async(batch, actor, critic, value)

            # Record best loss
            if loss_info[loss_key] < best_loss:
                best_loss = loss_info[loss_key]
                count = 0
                agent.actor.save(os.path.join('../experiments', agent.path, 'actor')) if actor else None
                agent.critic.save(os.path.join('../experiments', agent.path, 'critic')) if critic else None
                agent.value.save(os.path.join('../experiments', agent.path, 'value')) if value else None
            else:
                count += 1
                if count > 100:
                    agent.actor = agent.actor.load(os.path.join('../experiments', agent.path, 'actor')) if actor else agent.actor
                    agent.critic = agent.critic.load(os.path.join('../experiments', agent.path, 'critic')) if critic else agent.critic
                    agent.value = agent.value.load(os.path.join('../experiments', agent.path, 'value')) if value else agent.value
                    break

            # Log intermittently
            if epoch % 5 == 0:
                for key, val in loss_info.items():
                    if val.ndim == 0:
                        summary_writer.add_scalar(f'training/{key}', val, epoch)
                summary_writer.flush()

    """
    Other loss / metric recording aspects can go here
    """
    env = gymnasium.envs.make('LunarLander-v2')

    """
    import flax.linen as nn
    agent.actor = agent.actor.load('/Users/thomasfrost/Documents/Github/pygment/pygment/pygment/experiments/2023_8_29_215720/actor')
    agent.critic = agent.critic.load('/Users/thomasfrost/Documents/Github/pygment/pygment/pygment/experiments/2023_8_29_215720/critic')
    agent.value = agent.value.load('/Users/thomasfrost/Documents/Github/pygment/pygment/pygment/experiments/2023_8_29_215720/value')

    layer_output = []
    y = batch.states
    count = 0
    initializer = nn.initializers.he_normal()
    for layer in new_params['MLP_0'].values():
        y = y @ layer['kernel'] + layer['bias']
        y = nn.relu(y) if count != 2 else y
        layer_output.append(y)
    
        # This is the actual important code
        if count != 2:
            dead_units = jnp.quantile(y, 0.9, axis=0) == 0
            print(f'Dead units for layer {count} = {dead_units.sum()}')
            dead_units_idx = jnp.where(dead_units)[0]
            dead_shape = layer['kernel'][:, dead_units_idx].shape
            layer['kernel'][:, dead_units_idx] = initializer(jax.random.PRNGKey(42), dead_shape, jnp.float32)
            layer['bias'][dead_units_idx] = 0
    
        count += 1

        
        
    """