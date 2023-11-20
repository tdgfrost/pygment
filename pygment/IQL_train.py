import numpy as np
from gymnasium.envs import make as make_env
import os
import jax.numpy as jnp
from stable_baselines3.common.env_util import make_vec_env
import wandb
from scipy.stats import norm
import jax

# Set jax to CPU
# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)
# jax.config.update('jax_disable_jit', True)

# Define config file - could change to FLAGS at some point
config = {'seed': 123,
          'env_id': 'CartPole-v1',
          'step_delay': 2,
          'sync_steps': 20,
          'epochs': 100000,
          'early_stopping': jnp.array(1000),
          'batch_size': 10000,
          'expectile': 0.6,
          'dropout_rate': 0.5,
          'baseline_reward': 124,
          'n_episodes': 10000,
          'interval_probability': 0.25,
          # 'gaussian_confidence': 0.6,
          # 'confidence_quantile': None,
          'gamma': 0.99,
          'lr': 0.001,
          'alpha_soft_update': 1,
          'hidden_dims': (256, 256),
          'clipping': 1,
          'top_bar_coord': 1.2,  # 0.9,
          'bottom_bar_coord': 0.8,  # 0.5
          }

if __name__ == "__main__":
    from core.agent import IQLAgent
    from core.common import (load_data, progress_bar, alter_batch, filter_to_action,
                             calc_traj_discounted_rewards, move_to_gpu, filter_dataset, Batch)
    from core.evaluate import evaluate_envs, run_and_animate
    from core.envs import make_variable_env
    import argparse

    # Set the flags for expectile and soft_update
    parser = argparse.ArgumentParser()

    parser.add_argument('--expectile', type=float, default=config['expectile'])
    parser.add_argument('--soft_update', type=float, default=config['alpha_soft_update'])

    args = parser.parse_args()

    config['expectile'] = args.expectile
    config['alpha_soft_update'] = args.soft_update

    # Set whether to train and/or evaluate
    logging_bool = True
    evaluate_bool = False
    skip_value_training = True
    model_dir = './experiments/IQL/2023_11_18_161746'

    if logging_bool:
        wandb.init(
            project="CartPole-IQL-Gaussian-test",
            config=config,
        )

    # ============================================================== #
    # ========================= TRAINING =========================== #
    # ============================================================== #

    # Load static dataset
    print('Loading and processing dataset...')
    baseline_reward = config['baseline_reward']
    interval_probability = config['interval_probability']
    loaded_data = load_data(
        path=f"./offline_datasets/CartPole/{interval_probability}_probability/"
             f"dataset_reward_{baseline_reward}_{config['step_delay']}_steps_{config['n_episodes']}_episodes.pkl",
        # scale='standardise',
        gamma=config['gamma'])

    # Remove excess data
    loaded_data = loaded_data._asdict()
    for key in ['episode_rewards', 'next_actions', 'action_logprobs']:
        loaded_data[key] = None
    loaded_data = Batch(**loaded_data)

    # Add in normalisation
    # discounted_reward_mean = np.mean(loaded_data.discounted_rewards)
    # discounted_reward_std = np.std(loaded_data.discounted_rewards)

    # Start by defining the intervals between actions (both the current and next action)
    # intervals = the actual number of steps between actions
    intervals = np.array([len(traj) for traj in loaded_data.rewards])
    intervals = np.array([interval if not done else intervals.max()
                          for interval, done in zip(intervals.tolist(), loaded_data.dones.tolist())])
    intervals_unique = np.unique(intervals)
    mapping = {interval: idx for idx, interval in enumerate(intervals_unique)}
    len_actions = np.array([mapping[interval] for interval in intervals])
    next_len_actions = np.roll(len_actions, -1)

    intervals = jnp.array(intervals)
    len_actions = jnp.array(len_actions)
    next_len_actions = jnp.array(next_len_actions)

    # Calculate rewards
    rewards = jnp.array(calc_traj_discounted_rewards(loaded_data.rewards, config['gamma']))
    loaded_data = alter_batch(loaded_data, rewards=rewards, len_actions=len_actions, next_len_actions=next_len_actions,
                              intervals=intervals)
    del rewards

    # Move to GPU
    loaded_data = move_to_gpu(loaded_data, gpu_keys=['states', 'actions', 'discounted_rewards', 'episode_rewards',
                                                     'next_states', 'dones', 'action_logprobs', 'len_actions',
                                                     'rewards'])

    config['alpha_soft_update'] = jnp.array(config['alpha_soft_update'])

    # Make sure this matches with the desired dataset's extra_step metadata
    def extra_step_filter(x):
        # If tilted to the left
        if x[2] < 0:
            # with p == 0.25, delay by a further 5 steps (i.e., 6 total)
            if np.random.uniform() < interval_probability:
                return config['step_delay']
        # Otherwise, normal time steps (no delay)
        return 0

    # Train agent
    def train(data, model_dir=model_dir):
        # Create agent
        dummy_env = make_env(config['env_id'])

        agent = IQLAgent(observations=dummy_env.observation_space.sample(),
                         action_dim=dummy_env.action_space.n,
                         intervals_unique=intervals_unique,
                         opt_decay_schedule="cosine",
                         **config)

        if skip_value_training:
            agent.path = model_dir

        else:
            model_dir = os.path.join('./experiments/IQL', agent.path)
            agent.path = model_dir
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, 'config.txt'), 'w') as f:
                f.write(str(config))
                f.close()

        del dummy_env

        # Standardise the state inputs
        agent.standardise_inputs(data.states)

        if not skip_value_training:
            print('\n\n', '=' * 50, '\n', ' ' * 3, '\U0001F483' * 3, ' ' * 1, f'Training value networks',
                  ' ' * 2, '\U0001F483' * 3, '\n', '=' * 50, '\n')

            if logging_bool:
                # Keep track of the best loss values
                wandb.define_metric('actor_loss', summary='min')
                wandb.define_metric('average_value_loss', summary='min')
                wandb.define_metric('critic_loss', summary='min')
                wandb.define_metric('value_loss', summary='min')

            for epoch in range(config['epochs']):
                if epoch > 0 and epoch % 100 == 0:
                    print(f'\n\n{epoch} epochs complete!\n')
                progress_bar(epoch % 100, 100)
                batch, idxs = agent.sample(data,
                                           config['batch_size'])

                # Add Gaussian noise - remember last two positions are boolean
                """
                noise = np.random.normal(0, 0.01, size=batch.states.shape)
                noise[:, -2:] = 0
                batch = alter_batch(batch,
                                    states=batch.states + noise,
                                    next_states=batch.next_states + noise)
                """

                # Use TD learning for the value, average_value and critic networks
                gammas = np.ones(shape=config['batch_size']) * config['gamma']
                gammas = np.power(gammas, np.array(batch.intervals))

                # For Interval Value and Critic (from average value):
                agent.refresh_keys()
                next_state_values_avg = np.array(agent.target_value(batch.next_states,
                                                                    rngs={'dropout': agent.target_value_key})[1])

                discounted_rewards_for_interval_and_critic = (np.array(batch.rewards)
                                                              + gammas * next_state_values_avg * (1 - np.array(batch.dones))
                                                              )

                batch = alter_batch(batch, discounted_rewards=jnp.array(discounted_rewards_for_interval_and_critic),
                                    episode_rewards=None, next_states=None, next_actions=None, action_logprobs=None)

                # Perform the update step for interval value and critic networks
                value_loss_info = agent.update_async(batch,
                                                     value_loss_fn={'expectile': 0},
                                                     critic_loss_fn={'expectile': 0},
                                                     expectile=config['expectile'],
                                                     interval_value=True,
                                                     critic=True)

                value_loss_info['interval_value_loss'] = value_loss_info['value_loss']
                del value_loss_info['value_loss']

                # Then update for average network
                agent.refresh_keys()
                discounted_rewards_for_average = agent.interval_value(batch.states,
                                                                      rngs={'dropout': agent.interval_value_key})[1]
                discounted_rewards_for_average = filter_to_action(discounted_rewards_for_average, batch.len_actions)

                batch = alter_batch(batch,
                                    discounted_rewards=discounted_rewards_for_average)

                average_value_loss_info = agent.update_async(batch,
                                                             value_loss_fn={'mc_mse': 0},
                                                             average_value=True)

                average_value_loss_info['average_value_loss'] = average_value_loss_info['value_loss']
                del average_value_loss_info['value_loss']

                value_loss_info.update(average_value_loss_info)

                agent.sync_target(config['alpha_soft_update'])

                # Log intermittently
                if logging_bool:
                    # Log results
                    logged_results = {'gradient_step': epoch,
                                      'average_value_loss': value_loss_info['average_value_loss'],
                                      'critic_loss': value_loss_info['critic_loss'],
                                      'interval_value_loss': value_loss_info['interval_value_loss'],
                                      }

                    wandb.log(logged_results)

                if epoch % 100 == 0:
                    agent.target_value.save(os.path.join(model_dir, 'model_checkpoints/target_value'))
                    agent.critic.save(os.path.join(model_dir, 'model_checkpoints/critic'))
                    agent.interval_value.save(os.path.join(model_dir, 'model_checkpoints/interval_value'))

        else:
            agent.critic = agent.critic.load(os.path.join(model_dir, 'model_checkpoints/critic'))
            agent.interval_value = agent.interval_value.load(os.path.join(model_dir, 'model_checkpoints/interval_value'))
            agent.target_value = agent.target_value.load(os.path.join(model_dir, 'model_checkpoints/target_value'))

        # Perform MC dropout inference to assess uncertainty in values
        print('\n\n', '=' * 50, '\n', ' ' * 3, '\U0001F9D9' * 3, ' ' * 1, f'Calculating uncertainty',
              ' ' * 2, '\U0001F9D9' * 3, '\n', '=' * 50, '\n')

        agent.refresh_keys()
        state_values = np.array(agent.target_value(data.states,
                                                   rngs={'dropout': agent.target_value_key})[1])
        state_value_shape = state_values.shape
        state_values = state_values.reshape(1, *state_value_shape)

        critic_values = agent.critic(data.states,
                                     rngs={'dropout': agent.critic_key})[1]
        critic_values = np.array(filter_to_action(critic_values, data.actions))
        critic_values_shape = critic_values.shape
        critic_values = critic_values.reshape(1, *critic_values_shape)

        for _ in range(15):
            progress_bar(_, 15)
            agent.refresh_keys()
            state_values = np.concatenate((state_values,
                                           np.array(agent.target_value(data.states,
                                                                       rngs={'dropout': agent.target_value_key})[1]
                                                    ).reshape(1, *state_value_shape)))
            critic_values = np.concatenate((critic_values,
                                            np.array(filter_to_action(
                                                agent.critic(data.states,
                                                             rngs={'dropout': agent.critic_key})[1],
                                                data.actions)).reshape(1, *critic_values_shape)))

        state_values_mean = np.mean(state_values, axis=0)
        state_values_std = np.std(state_values, axis=0)
        critic_values_mean = np.mean(critic_values, axis=0)
        critic_values_std = np.std(critic_values, axis=0)

        advantages = critic_values_mean - state_values_mean
        advantages_std = np.sqrt(state_values_std ** 2 + critic_values_std ** 2)

        # Filter for top x% of "good" actions
        """
        cdf = 1 - norm.cdf(0, advantages, advantages_std)
        if 'gaussian_confidence' in config.keys() and config['gaussian_confidence'] is not None:
            filter_point = config['gaussian_confidence']
        else:
            filter_point = np.quantile(cdf, config['confidence_quantile'])
        """
        # filter_bool = cdf > filter_point
        filter_bool = advantages > 0

        if np.sum(filter_bool) < config['batch_size']:
            return agent

        data = filter_dataset(data, filter_bool,
                              target_keys=['states', 'actions'])

        print('\n\n', '=' * 50, '\n', ' ' * 3, '\U0001F483' * 3, ' ' * 1, f'Training actor',
              ' ' * 2, '\U0001F483' * 3, '\n', '=' * 50, '\n')

        # Now we start training the actor
        for epoch in range(config['epochs']):
            if epoch > 0 and epoch % 100 == 0:
                print(f'\n\n{epoch} epochs complete!\n')
            progress_bar(epoch % 100, 100)
            batch, idxs = agent.sample(data,
                                       config['batch_size'])

            batch = alter_batch(batch, discounted_rewards=None, episode_rewards=None, next_states=None,
                                next_actions=None, action_logprobs=None, len_actions=None, rewards=None,
                                next_len_actions=None, intervals=None, dones=None)

            actor_loss_info = agent.update_async(batch,
                                                 actor_loss_fn={'clone': 0},
                                                 actor=True)

            episode_rewards = None
            if epoch % config['sync_steps'] == 0:
                episode_rewards = evaluate_envs(agent, make_vec_env(lambda: make_variable_env(config['env_id'],
                                                                                              fn=extra_step_filter),
                                                                    n_envs=1),
                                                verbose=False)

            # Log intermittently
            if logging_bool:
                # Log results
                logged_results = {'gradient_step': epoch,
                                  'actor_loss': actor_loss_info['actor_loss'],
                                  }
                if epoch % config['sync_steps'] == 0:
                    logged_results.update({'mean_reward': np.mean(episode_rewards)})

                wandb.log(logged_results)

            if epoch % 100 == 0:
                # Save each model
                agent.actor.save(os.path.join(model_dir, 'model_checkpoints/actor'))

        # Evaluate agent
        n_envs = 1000
        print('\n\n', '=' * 50, '\n', ' ' * 3, '\U0001F514' * 3, ' ' * 1, f'Evaluating network', ' ' * 2,
              '\U0001F514' * 3, '\n', '=' * 50)
        episode_rewards = evaluate_envs(agent, make_vec_env(lambda: make_variable_env(config['env_id'],
                                                                                      fn=extra_step_filter),
                                                            n_envs=n_envs))
        print(f'\nMedian reward: {np.median(episode_rewards)}')
        print(f'Mean reward: {np.mean(episode_rewards)}')

        with open(os.path.join(model_dir, 'rewards.txt'), 'w') as f:
            f.write(f'Baseline reward: {baseline_reward}\n')
            f.write(f'Median reward: {np.median(episode_rewards)}\n')
            f.write(f'Mean reward: {np.mean(episode_rewards)}\n')
            f.close()

        return agent


    # Run the train script
    agent = train(loaded_data)

    # ============================================================== #
    # ======================== EVALUATION ========================== #
    # ============================================================== #

    if evaluate_bool:
        dummy_env = make_env(config['env_id'])

        agent = IQLAgent(observations=dummy_env.observation_space.sample(),
                         action_dim=dummy_env.action_space.n,
                         intervals_unique=intervals_unique,
                         opt_decay_schedule="cosine",
                         **config)

        del dummy_env

        model_dir = os.path.join('./experiments/IQL', agent.path)
        agent.actor = agent.actor.load(os.path.join(model_dir, 'model_checkpoints/actor'))

        max_episode_steps = 1000
        envs_to_evaluate = 1000

        print('\n\n', '=' * 50, '\n', ' ' * 3, '\U0001F514' * 3, ' ' * 1, f'Evaluating network', ' ' * 2,
              '\U0001F514' * 3, '\n', '=' * 50)
        results = evaluate_envs(agent, make_vec_env(lambda: make_variable_env(config['env_id'],
                                                                              fn=extra_step_filter),
                                                    n_envs=envs_to_evaluate))
        print(f'\nMedian reward: {np.median(results)}')
        print(f'\nMean reward: {np.mean(results)}')

        # Animate the agent's performance
        print('\n\n', '=' * 50, '\n', ' ' * 3, '\U0001F4FA' * 3, ' ' * 1, f'Generating gifs', ' ' * 2,
              '\U0001F4FA' * 3, '\n', '=' * 50)
        env = make_variable_env(config['env_id'], fn=extra_step_filter, render_mode='rgb_array')
        run_and_animate(agent, env, runs=20, directory=os.path.join(model_dir, 'gifs'), **config)
