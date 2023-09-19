from tqdm import tqdm
import numpy as np
import gymnasium
from gymnasium.envs import make as make_env
from tensorboardX import SummaryWriter
import os
import jax
import jax.numpy as jnp
from stable_baselines3.common.env_util import make_vec_env

# Set jax to CPU
# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)
# jax.config.update('jax_disable_jit', True)

# Define config file - could change to FLAGS at some point
config = {'seed': 123,
          'epochs': int(1e6),
          'batch_size': int(1e5),
          'expectile': 0.5,
          'gamma': 0.9999,
          'actor_lr': 5e-3,
          'value_lr': 5e-3,
          'critic_lr': 5e-3,
          'hidden_dims': (512, 512),
          'clipping': 1,
          }


if __name__ == "__main__":
    from core.agent import IQLAgent
    from core.common import load_data, progress_bar, Batch
    from update.loss import expectile_loss, iql_loss, mse_loss
    from core.evaluate import evaluate_envs, run_and_animate
    from core.envs import make_variable_env

    # Set whether to train and/or evaluate
    train = True
    evaluate = True

    # Create environment
    dummy_env = make_env('LunarLander-v2')

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
    agent = IQLAgent(observations=dummy_env.observation_space.sample(),
                     action_dim=dummy_env.action_space.n,
                     dropout_rate=None,
                     opt_decay_schedule="cosine",
                     **config)

    del dummy_env

    # Prepare logging tensorboard
    summary_writer = SummaryWriter('../experiments/tensorboard/current',
                                   write_to_disk=True)
    os.makedirs('../experiments/tensorboard/current/', exist_ok=True)

    # Train agent
    if train:
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
                    if count > 1000:
                        agent.actor = agent.actor.load(
                            os.path.join('../experiments', agent.path, 'actor')) if actor else agent.actor
                        agent.critic = agent.critic.load(
                            os.path.join('../experiments', agent.path, 'critic')) if critic else agent.critic
                        agent.value = agent.value.load(
                            os.path.join('../experiments', agent.path, 'value')) if value else agent.value
                        break

                # Log intermittently
                if epoch % 5 == 0:
                    for key, val in loss_info.items():
                        if key == 'layer_outputs':
                            continue
                        if val.ndim == 0:
                            summary_writer.add_scalar(f'training/{key}', val, epoch)
                    summary_writer.flush()

    """
    Time to evaluate!
    """
    if evaluate:
        filename = agent.path
        agent.actor = agent.actor.load(os.path.join('../experiments', f'{filename}', 'actor'))
        agent.critic = agent.critic.load(os.path.join('../experiments', f'{filename}', 'critic'))
        agent.value = agent.value.load(os.path.join('../experiments', f'{filename}', 'value'))

        max_episode_steps = 1000
        envs_to_evaluate = 1000

        results = evaluate_envs(agent, make_vec_env('LunarLander-v2', n_envs=envs_to_evaluate))
        print(f'\nMedian reward: {np.median(results)}')

