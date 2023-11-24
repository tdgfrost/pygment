import numpy as np
from gymnasium.envs import make as make_env
import os
import jax.numpy as jnp
from stable_baselines3.common.env_util import make_vec_env
import wandb
from math import ceil

# Set jax to CPU
# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)
# jax.config.update('jax_disable_jit', True)

# Define config file - could change to FLAGS at some point
config = {'seed': 123,
          'epochs': 100000,
          'env_id': 'LunarLander-v2',
          'early_stopping': jnp.array(1000),
          'batch_size': 10000,
          'step_delay': 11,
          'sync_steps': 20,
          'expectile': 0.5,
          'baseline_reward': 45,
          'n_episodes': 10000,
          'interval_probability': 0.25,
          'top_actions_quantile': 0.5,
          'filter_point': 0,
          'gamma': 0.99,
          'actor_lr': 0.001,
          'value_lr': 0.001,
          'critic_lr': 0.001,
          'alpha_soft_update': 1,
          'hidden_dims': (256, 256),
          'clipping': 1
          }

if __name__ == "__main__":
    from core.agent import IQLAgent
    from core.common import (load_data, progress_bar, alter_batch, Batch, filter_to_action,
                             calc_traj_discounted_rewards, move_to_gpu, filter_dataset)
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

    if logging_bool:
        wandb.init(
            project="LunarLander-25pct-R-45",
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
        path=f"./offline_datasets/LunarLander/{interval_probability}_probability/"
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
    """
    INCLUDE THIS??
    intervals = np.array([interval if not done else intervals.max()
                      for interval, done in zip(intervals.tolist(), loaded_data.dones.tolist())])
    """
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

    # config['gamma'] = jnp.array(config['gamma'])
    config['alpha_soft_update'] = jnp.array(config['alpha_soft_update'])


    # Make sure this matches with the desired dataset's extra_step metadata
    def extra_step_filter(x):
        # If tilted left
        if x[2] < 0:
            # Slow zone
            if np.random.uniform() < interval_probability:
                return config['step_delay']
        # Otherwise, normal time steps (no delay)
        return 0


    # Train agent
    def train(data):
        # Create agent
        dummy_env = make_env(config['env_id'])

        agent = IQLAgent(observations=dummy_env.observation_space.sample(),
                         action_dim=dummy_env.action_space.n,
                         intervals_unique=intervals_unique,
                         opt_decay_schedule="cosine",
                         **config)

        model_dir = os.path.join('./experiments/IQL', agent.path)
        agent.path = model_dir
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'config.txt'), 'w') as f:
            f.write(str(config))
            f.close()

        del dummy_env

        # Standardise the state inputs
        agent.standardise_inputs(data.states)

        print('\n\n', '=' * 50, '\n', ' ' * 3, '\U0001F483' * 3, ' ' * 1, f'Training network',
              ' ' * 2, '\U0001F483' * 3, '\n', '=' * 50, '\n')

        total_training_steps = jnp.array(0)

        if logging_bool:
            # Keep track of the best loss values
            wandb.define_metric('actor_loss', summary='min')
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
            # Use TD learning for the value and critic networks
            gammas = np.ones(shape=len(batch.rewards)) * config['gamma']
            gammas = np.power(gammas, np.array(batch.intervals))

            next_state_values = np.array(agent.target_value(batch.next_states)[1])
            discounted_rewards = np.array(batch.rewards) + gammas * next_state_values * (1 - np.array(batch.dones))

            batch = alter_batch(batch, discounted_rewards=jnp.array(discounted_rewards), next_states=None,
                                dones=None, intervals=None, rewards=None)

            # Perform the update step for interval value and critic networks
            value_loss_info = agent.update_async(batch,
                                                 value_loss_fn={'expectile': 0},
                                                 critic_loss_fn={'expectile': 0},
                                                 expectile=config['expectile'],
                                                 value=True,
                                                 critic=True)

            # Do a partial sync with the target network
            agent.sync_target(config['alpha_soft_update'])

            # Log intermittently
            if logging_bool:
                # Log results
                logged_results = {'training_step': total_training_steps,
                                  'gradient_step': epoch,
                                  'value_loss': value_loss_info['value_loss'],
                                  'critic_loss': value_loss_info['critic_loss'],
                                  }

                wandb.log(logged_results)

        # And train the actor
        total_training_steps = jnp.array(0)

        # Calculate the advantages
        advantages = []
        step_size = int(5e5)
        for i in range(ceil(data.states.shape[0] / step_size)):
            progress_bar(i, ceil(data.states.shape[0] / step_size))
            idx = slice(i * step_size, (i+1)*step_size, 1)
            state_values = agent.target_value(data.states[idx])[1]

            critic_values = jnp.minimum(*agent.critic(data.states[idx])[1])
            critic_values = filter_to_action(critic_values, data.actions[idx])

            advantages += [critic_values - state_values]

        # Filter for top half of actions
        if 'filter_point' not in config.keys():
            filter_point = np.quantile(np.array(advantages), config['top_actions_quantile'])
        else:
            filter_point = config['filter_point']

        if np.sum(np.array(advantages) > filter_point) < config['batch_size']:
            return agent

        data = filter_dataset(data, np.array(advantages) > filter_point,
                              target_keys=['states', 'actions'])

        data = alter_batch(data, discounted_rewards=None, next_states=None, dones=None, intervals=None,
                           rewards=None, len_actions=None, next_len_actions=None)

        for epoch in range(config['epochs']):
            if epoch > 0 and epoch % 100 == 0:
                print(f'\n\n{epoch} epochs complete!\n')

            progress_bar(epoch % 100, 100)
            batch, idxs = agent.sample(data,
                                       config['batch_size'])

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
                logged_results = {'training_step': total_training_steps,
                                  'gradient_step': epoch,
                                  'actor_loss': actor_loss_info['actor_loss'],
                                  }
                if epoch % config['sync_steps'] == 0:
                    logged_results.update({'mean_reward': np.mean(episode_rewards)})

                wandb.log(logged_results)

            if epoch % 100 == 0:
                # Save each model
                agent.actor.save(os.path.join(model_dir, 'model_checkpoints/actor'))
                agent.target_value.save(os.path.join(model_dir, 'model_checkpoints/target_value'))
                agent.critic.save(os.path.join(model_dir, 'model_checkpoints/critic'))
                agent.value.save(os.path.join(model_dir, 'model_checkpoints/value'))

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
