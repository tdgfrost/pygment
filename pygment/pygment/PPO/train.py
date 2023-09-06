from tqdm import tqdm
import numpy as np
import gymnasium
from tensorboardX import SummaryWriter
import os
import jax
import jax.numpy as jnp
from stable_baselines3.common.env_util import make_vec_env
from sklearn.cluster import KMeans

# Set jax to CPU
# jax.config.update('jax_platform_name', 'cpu')
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
          'hidden_dims': (512, 512),
          'clipping': 1,
          }

if __name__ == "__main__":
    from agent import IQLAgent
    from common import load_data, Batch, progress_bar
    from envs import VariableTimeSteps

    # Set whether to train and/or evaluate
    train = True
    evaluate = True

    # Set the number of clusters to use
    n_clusters = 10

    # Load static dataset (dictionary)
    data = load_data(path='../samples/GenerateStaticDataset/LunarLander/140 reward',
                     scale='standardise',
                     gamma=config['gamma'])['state']

    # Generate clustering function
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=123).fit(data[:1e6])

    # Create environment and wrap in time-delay wrapper
    env = gymnasium.envs.make('LunarLander-v2', max_episode_steps=1000)
    env = VariableTimeSteps(env, fn=lambda x: kmeans.predict(x.reshape(1, -1))[0] if len(x.shape) == 1
                                                                                  else kmeans.predict(x))

    # Create agent
    pass

    # Prepare logging tensorboard
    summary_writer = SummaryWriter('../experiments/tensorboard/current',
                                   write_to_disk=True)
    os.makedirs('../experiments/tensorboard/current/', exist_ok=True)

    # Train agent
    if train:
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


        def make_env():
            env = gymnasium.envs.make('LunarLander-v2', max_episode_steps=max_episode_steps)
            return env


        def evaluate_envs(nodes=10):
            """
            Evaluate the agent across vectorised episodes.

            :param nodes: number of episodes to evaluate.
            :return: array of total rewards for each episode.
            """
            envs = make_vec_env(make_env, n_envs=nodes)

            # Initial parameters
            key = jax.random.PRNGKey(123)
            states = envs.reset()
            dones = np.array([False for _ in range(nodes)])
            idxs = np.array([i for i in range(nodes)])
            all_rewards = np.array([0. for _ in range(nodes)])
            step = 0

            while not dones.all():
                step += 1
                progress_bar(step, max_episode_steps)
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