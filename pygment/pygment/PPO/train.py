import os

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.cluster import KMeans
import wandb
from tqdm import tqdm
from stable_baselines3.common.env_util import make_vec_env

# Set jax to CPU
# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)
# jax.config.update('jax_disable_jit', True)

# Define config file - could change to FLAGS at some point
config = {'seed': 123,
          'epochs': int(1e6),
          'steps': 5000,
          'batch_size': 32,
          'n_envs': 20,
          'gamma': 0.99,
          'actor_lr': 0.001,
          'value_lr': 0.001,
          'hidden_dims': (64, 64),
          'clipping': 1,
          }

if __name__ == "__main__":
    from agent import PPOAgent
    from common import load_data, progress_bar, Batch, shuffle_split_batch, alter_batch, downsample_batch
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
    extra_step_filter = lambda x: 10 if 0.8 < x[1] < 1.2 else 0
    env = make_env(fn=extra_step_filter)
    envs = make_vec_env(lambda: make_env(fn=extra_step_filter), n_envs=config['n_envs'])

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
        project="PPO-VariableTimeSteps",
        config=config,
    )

    # Train agent
    if train:
        # Generate initial log variables + random key
        best_reward = -1000
        key = jax.random.PRNGKey(123)
        count = 0

        # Generate initial batch
        batch, key = sampler(agent, key=key)

        # Remove anything not needed for jitted training
        excess_data = {}
        removed_data = {}
        for item in ['rewards', 'episode_rewards', 'next_actions', 'next_states', 'dones']:
            excess_data[item] = batch._asdict()[item]
            removed_data[item] = None

        batch = alter_batch(batch, **removed_data)

        # Select a random subset of the batch
        batch, key = downsample_batch(batch, key, config['steps'])

        # Train agent
        for epoch in tqdm(range(config['epochs'])):
            actor_loss = 0
            critic_loss = 0

            iteration = 0
            update_iters = 4
            for update_iter in range(update_iters):
                # Every iteration, the advantage should be re-calculated
                batch_state_values = np.array(agent.value(batch.states)[1])
                advantages = batch.discounted_rewards - batch_state_values
                advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-8)

                batch = alter_batch(batch, advantages=advantages)

                # Shuffle the batch
                shuffled_batch = shuffle_split_batch(batch,
                                                     steps=config['steps'],
                                                     batch_size=config['batch_size'])
                # Iterate through each sample in the batch
                for sample in shuffled_batch:
                    iteration += 1
                    progress_bar(iteration, len(batch.actions) // config['batch_size'] * update_iters)
                    # Update the agent
                    loss_info = agent.update(sample)

                    # Update the loss
                    actor_loss += loss_info['actor_loss'].item()
                    critic_loss += loss_info['value_loss'].item()

            # Reset the jax key
            key = jax.random.split(key, 1)[0]

            # Generate the next batch using the updated agent
            batch, key = sampler(agent, key=key)

            # Remove anything not needed for training
            excess_data = {}
            removed_data = {}
            for item in ['rewards', 'episode_rewards', 'next_actions', 'next_states', 'dones']:
                excess_data[item] = batch._asdict()[item]
                removed_data[item] = None

            batch = alter_batch(batch, **removed_data)

            # Select a random subset of the batch
            batch, key = downsample_batch(batch, key, config['steps'])

            # Check the value function is training correctly
            batch_state_values = np.array(agent.value(batch.states)[1])
            tmp_idx = np.random.permutation([i for i in range(len(batch.discounted_rewards))])[:5]
            print('\n\nDiscounted rewards: ', batch.discounted_rewards[tmp_idx])
            print('Value function: ', batch_state_values[tmp_idx])
            print('Episode rewards: ', excess_data['episode_rewards'].mean(), '\n')

            # Calculate the average reward (for logging purposes)
            average_reward = excess_data['episode_rewards'].mean()

            # Checkpoint the model
            if int(average_reward) > best_reward:
                best_reward = int(average_reward)
                # count = 0
                agent.actor.save(os.path.join('../experiments',
                                              agent.path, f'actor_{best_reward}'))  # if actor else None
                agent.value.save(os.path.join('../experiments',
                                              agent.path, f'value_{best_reward}'))  # if value else None
            """
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

            if best_reward > 250:
                agent.actor.save(os.path.join('../experiments',
                                              agent.path, f'actor_best'))  # if actor else None
                agent.value.save(os.path.join('../experiments',
                                              agent.path, f'value_best'))
                break

    """
    Time to evaluate!
    """
    if evaluate:
        filename = agent.path
        agent.actor = agent.actor.load(os.path.join('../experiments', f'{filename}', f'actor_best'))
        agent.value = agent.value.load(os.path.join('../experiments', f'{filename}', f'value_best'))

        max_episode_steps = 1000
        envs_to_evaluate = 1000


        def evaluate_envs(policy, envs):
            """
            Evaluate the agent across vectorised episodes.

            :param policy: policy to evaluate.
            :param envs: envs to evaluate.
            :return: array of total rewards for each episode.
            """
            nodes = envs.num_envs
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

        results = evaluate_envs(agent, envs=envs)
        print(f'\nMedian reward: {np.median(results)}')
        input('Ready to render environments? Press enter when ready')
        key = jax.random.PRNGKey(123)
        for _ in range(3):
            env = make_env(fn=extra_step_filter)
            state = env.reset()
            done, prem_done = False, False
            while not done and not prem_done:
                action = agent.sample_action(state, key)
                state, _, done, prem_done, _ = env.step(action)

            env.render()
            env.close()
