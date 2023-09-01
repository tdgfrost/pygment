from tqdm import tqdm
import numpy as np
import gymnasium
from tensorboardX import SummaryWriter
import os
import jax
import jax.numpy as jnp
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# Set jax to CPU
jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)
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


def progress_bar(iteration, total_iterations):
    """
    Print a progress bar to the console
    :param iteration: current iteration
    :param total_iterations: total number of iterations
    :return: None
    """
    bar_length = 30

    percent = iteration / total_iterations
    percent_complete = int(percent * 100)

    progress = int(percent * bar_length)
    progress = '[' + '#' * progress + ' ' * (bar_length - progress) + ']'

    print(f'\r{progress} {percent_complete}% complete', end='', flush=True)
    return


if __name__ == "__main__":
    from agent import IQLAgent
    from common import load_data, Batch

    # Set whether to train and/or evaluate
    train = False
    evaluate = True

    # Create environment
    env = gymnasium.envs.make('LunarLander-v2', max_episode_steps=1000)

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
                     action_dim=env.action_space.n,
                     dropout_rate=None,
                     opt_decay_schedule="cosine",
                     **config)

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
        agent.actor = agent.actor.load(os.path.join('../experiments/experiment_3', 'actor'))
        agent.critic = agent.critic.load(os.path.join('../experiments/experiment_3', 'critic'))
        agent.value = agent.value.load(os.path.join('../experiments/experiment_3', 'value'))

        max_episode_steps = 1000
        envs_to_evaluate = 1000

        def make_env():
            env = gymnasium.envs.make('LunarLander-v2', max_episode_steps=max_episode_steps)
            return env

        def evaluate_envs(nodes=10):
            envs = make_vec_env(make_env, n_envs=nodes)

            # Initial parameters
            key = jax.random.PRNGKey(123)
            states = envs.reset()
            dones = np.array([False for _ in range(nodes)])
            idxs = np.array([i for i in range(nodes)])
            all_rewards = np.array([0. for _ in range(nodes)])
            count = 0
            while not dones.all():
                count += 1
                progress_bar(count, max_episode_steps)
                # Step through environments
                actions = np.array(agent.sample_action(states, key))
                states, rewards, new_dones, prem_dones = envs.step(actions)

                # Update finished environments
                prem_dones = np.array([d['TimeLimit.truncated'] for d in prem_dones])
                dones[idxs] = np.any((new_dones, prem_dones), axis=0)[idxs]

                # Update rewards
                all_rewards[idxs] += np.array(rewards)[idxs]

                # Update remaining parameters
                idxs = np.where(~dones)[0]
                states = np.array(states)
                key = jax.random.split(key, num=1)[0]

            return all_rewards

        results = evaluate_envs(envs_to_evaluate)
        print(f'\nMedian reward: {np.median(results)}')
