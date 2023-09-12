import os

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.cluster import KMeans
import wandb
from tqdm import tqdm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

# Set jax to CPU
# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)
# jax.config.update('jax_disable_jit', True)

# Define config file - could change to FLAGS at some point
config = {'seed': 123,
          'epochs': int(1e6),
          'steps': 1000,
          'batch_size': 64,
          'n_envs': 512,
          'gamma': 0.999,
          'actor_lr': 5e-3,
          'value_lr': 5e-3,
          'hidden_dims': (256, 256),
          'clipping': 1,
          }

if __name__ == "__main__":
    from agent import PPOAgent
    from common import load_data, progress_bar, Batch, shuffle_batch, alter_batch
    from envs import VariableTimeSteps, EpisodeGenerator

    # Set whether to train and/or evaluate
    train = True
    evaluate = True

    # Set the number of clusters to use
    n_clusters = 5

    # Load static dataset (dictionary)
    data = load_data(
        path='/Users/thomasfrost/Documents/Github/pygment/pygment/pygment/samples/GenerateStaticDataset/LunarLander/140 reward',
        scale='standardise',
        gamma=config['gamma'])['state']

    # Generate clustering function
    """
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=123,
                    n_init='auto').fit(data[:int(1e6)])
    """
    # Create environment and wrap in time-delay wrapper
    def make_env(fn=None):
        environment = VariableTimeSteps(gymnasium.envs.make('LunarLander-v2', max_episode_steps=500),
                                        fn=fn)
        return environment


    # env = make_env(lambda x: kmeans.predict(x.reshape(1, -1))[0] if len(x.shape) == 1 else kmeans.predict(x))
    env = make_env(fn=None)
    envs = make_vec_env(lambda: make_env(fn=None), n_envs=config['n_envs'])

    # Create episode generator
    sampler = EpisodeGenerator(envs, gamma=config['gamma'])

    # Create agent
    agent = PPOAgent(observations=env.observation_space.sample(),
                     action_dim=env.action_space.n,
                     dropout_rate=None,
                     opt_decay_schedule="cosine",
                     **config)

    # Prepare logging
    wandb.init(
        project="PPO",
        config=config,
    )

    # Train agent
    if train:
        # Generate initial log variables + random key
        best_actor_loss = jnp.inf
        key = jax.random.PRNGKey(123)
        count = 0

        # Generate initial batch
        batch, key = sampler(agent, key=key)

        # Train agent
        for epoch in tqdm(range(config['epochs'])):
            actor_loss = 0
            critic_loss = 0

            for update_iter in range(5):
                # Every iteration, the advantage should be re-calculated
                batch_state_values = [np.array(agent.value(np.array(ep))[1]) for ep in batch.states]
                advantages = [ep_r - ep_v for ep_r, ep_v in zip(batch.discounted_rewards, batch_state_values)]
                adv_mean = np.concatenate(advantages).mean()
                adv_std = np.maximum(np.concatenate(advantages).std(), 1e-8)
                advantages = [(adv - adv_mean) / adv_std for adv in advantages]

                batch = alter_batch(batch, advantages=advantages)

                # Shuffle the batch
                shuffled_batch = shuffle_batch(batch,
                                               key,
                                               steps=config['steps'],
                                               batch_size=config['batch_size'])
                # Iterate through each sample in the batch
                for sample in shuffled_batch:
                    # Update the agent
                    loss_info = agent.update(sample)

                    # Update the loss
                    actor_loss += loss_info['actor_loss'].item()
                    critic_loss += loss_info['value_loss'].item()

            # Reset the jax key
            key = jax.random.split(key, 1)[0]

            # Generate the next batch using the updated agent
            batch, key = sampler(agent, key=key)

            # Check the value function is training correctly
            disc_r = np.array([r for ep in batch.discounted_rewards for r in ep])
            v = np.array(agent.value(np.array([s for ep in batch.states for s in ep]))[1])
            tmp_idx = np.random.permutation([i for i in range(len(disc_r))])[:5]
            print('\nDiscounted rewards: ', disc_r[tmp_idx])
            print('Value function: ', v[tmp_idx])

            # Calculate the average reward (for logging purposes)
            average_reward = np.mean(batch.episode_rewards)

            """
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
            """
            # Log results
            wandb.log({'actor_loss': actor_loss,
                       'critic_loss': critic_loss,
                       'episode_reward': average_reward})

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


        def evaluate_envs(policy, nodes=10):
            """
            Evaluate the agent across vectorised episodes.

            :param policy: policy to evaluate.
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
                actions = np.array(policy.sample_action(states, key))
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


        results = evaluate_envs(agent, envs_to_evaluate)
        print(f'\nMedian reward: {np.median(results)}')
