import os

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.cluster import KMeans
import wandb
from tqdm import tqdm
from stable_baselines3.common.env_util import make_vec_env
from matplotlib.patches import Rectangle

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
          'top_bar_coord': 1.2,  # 0.9,
          'bottom_bar_coord': 0.8,  # 0.5
          }

if __name__ == "__main__":
    from agent import PPOAgent
    from common import (load_data, progress_bar, shuffle_split_batch, alter_batch, downsample_batch, evaluate_envs,
                        animate_blocked_environment)
    from envs import VariableTimeSteps, EpisodeGenerator

    # Set whether to train and/or evaluate
    train = False
    evaluate = True

    # Load static dataset (dictionary)
    """
    data = load_data(
        path='/Users/thomasfrost/Documents/Github/pygment/pygment/pygment/samples/GenerateStaticDataset/LunarLander/140 reward',
        scale='standardise',
        gamma=config['gamma'])['state']
    """
    # Create agent
    env = gymnasium.envs.make('LunarLander-v2', max_episode_steps=500)
    agent = PPOAgent(observations=env.observation_space.sample(),
                     action_dim=env.action_space.n,
                     dropout_rate=None,
                     opt_decay_schedule="cosine",
                     **config)
    del env

    # Create environment and wrap in time-delay wrapper
    def make_env(fn=None, max_episode_steps=500, render_mode=None):
        environment = VariableTimeSteps(gymnasium.envs.make('LunarLander-v2', max_episode_steps=max_episode_steps,
                                                            render_mode=render_mode),
                                        fn=fn)
        return environment


    # Generate clustering function
    """
    # Set the number of clusters to use
    n_clusters = 5

    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=123,
                    n_init='auto').fit(data[:int(1e6)])
    """

    """
    env = make_env(lambda x: kmeans.predict(x.reshape(1, -1))[0] if len(x.shape) == 1 else kmeans.predict(x))
    """
    def extra_step_filter(x):
        # If in rectangle
        if config['bottom_bar_coord'] < x[1] < config['top_bar_coord']:
            # with p == 0.05, delay by 20 steps
            if np.random.uniform() < 0.05:
                return 20
        # Otherwise, normal time steps (no delay)
        return 0

    envs = make_vec_env(lambda: make_env(fn=extra_step_filter), n_envs=config['n_envs'])

    # Train agent
    if train:
        # Prepare logging
        wandb.init(
            project="PPO-VariableTimeSteps",
            config=config,
        )

        # Create episode generator
        sampler = EpisodeGenerator(envs, gamma=config['gamma'])

        # Generate initial log variables + random key
        best_reward = -1000
        random_key = jax.random.PRNGKey(123)
        count = 0

        # Generate initial batch
        batch, random_key = sampler(agent, key=random_key)

        # Remove anything not needed for jitted training
        excess_data = {}
        removed_data = {}
        for item in ['rewards', 'episode_rewards', 'next_actions', 'next_states', 'dones']:
            excess_data[item] = batch._asdict()[item]
            removed_data[item] = None

        batch = alter_batch(batch, **removed_data)

        # Select a random subset of the batch
        batch, random_key = downsample_batch(batch, random_key, config['steps'])

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
            random_key = jax.random.split(random_key, 1)[0]

            # Generate the next batch using the updated agent
            batch, random_key = sampler(agent, key=random_key)

            # Remove anything not needed for training
            excess_data = {}
            removed_data = {}
            for item in ['rewards', 'episode_rewards', 'next_actions', 'next_states', 'dones']:
                excess_data[item] = batch._asdict()[item]
                removed_data[item] = None

            batch = alter_batch(batch, **removed_data)

            # Select a random subset of the batch
            batch, random_key = downsample_batch(batch, random_key, config['steps'])

            # Check the value function is training correctly
            batch_state_values = np.array(agent.value(batch.states)[1])
            tmp_idx = np.random.permutation([i for i in range(len(batch.discounted_rewards))])[:5]
            print('\n\nDiscounted rewards: ', batch.discounted_rewards[tmp_idx])
            print('Value function: ', batch_state_values[tmp_idx])
            print('Episode rewards: ', np.median(excess_data['episode_rewards']), '\n')

            # Calculate the average reward (for logging purposes)
            average_reward = np.median(excess_data['episode_rewards'])

            # Checkpoint the model
            if int(average_reward) > best_reward:
                print('Evaluating performance...')
                results = evaluate_envs(agent,
                                        environments=make_vec_env(lambda: make_env(fn=extra_step_filter),
                                                                  n_envs=1000))
                average_reward = np.median(results)
                print('\n\n', '='*50, f'\nMedian reward: {np.median(results)}, Best reward: {best_reward}\n', '='*50, '\n')
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
        # Load the best agent
        filename = './Experiment_2/model_checkpoints'  # agent.path
        agent.actor = agent.actor.load(os.path.join('../experiments', f'{filename}', f'actor_best'))  # f'actor_best'))
        agent.value = agent.value.load(os.path.join('../experiments', f'{filename}', f'value_best'))  # f'value_best'))

        # Define the number of environments to evaluate over
        envs_to_evaluate = 1000

        # Create the vectorised set of environments
        envs = make_vec_env(lambda: make_env(fn=extra_step_filter), n_envs=envs_to_evaluate)

        # Calculate the median reward
        results = evaluate_envs(agent, environments=envs)
        print(f'\nMedian reward: {np.median(results)}')

        # Render the environment (optional) with a plotted 'slow zone' rectangle
        """
        render_envs = bool(input('Render? Type anything if yes, or blank if no'))
        """
        # Set the random key
        random_key = jax.random.PRNGKey(123)

        # Create the environment
        env = make_env(render_mode='rgb_array', fn=extra_step_filter)

        # Set the parameters for the Rectangle to be plotted
        top = config['top_bar_coord']
        bottom = config['bottom_bar_coord']
        img_grad = 300 / 1.5
        width = (top - bottom) * img_grad

        # Then render as many times as required
        # count = 0
        # while render_envs:
        for count in range(20):
            progress_bar(count, 20)
            # count += 1

            # Define the empty list of frames
            img_arrays = []

            # Define the random key
            random_key = jax.random.split(random_key, num=1)[0]

            # Iterate over the environment, saving all the frames as rgb_arrays
            state, _ = env.reset()
            done, prem_done = False, False
            img_arrays.append(env.render())
            while not done and not prem_done:
                action = agent.sample_action(state, random_key)[0]
                state, _, done, prem_done, info = env.step(action)
                img_arrays.extend(info['render_arrays'])

            # Define the Rectangle and animate the environment with it
            rect = Rectangle((0, (300 - img_grad * top)),
                             600,
                             width,
                             linewidth=1,
                             edgecolor='r',
                             facecolor='none')

            animate_blocked_environment(img_arrays,
                                        #os.path.join('../experiments', f'{filename}', f'test_{count}.gif'),
                                        os.path.join('../experiments', f'test_{count}.gif'),
                                        patch=rect,
                                        fps=env.metadata['render_fps'])
            # render_envs = bool(input('Render? Type anything if yes, or blank if no'))

        # Close the environment
        env.close()
