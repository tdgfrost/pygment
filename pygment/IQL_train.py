import jax
import numpy as np
from gymnasium.envs import make as make_env
import os
import jax.numpy as jnp
from stable_baselines3.common.env_util import make_vec_env
import wandb

# Set jax to CPU
jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)
# jax.config.update('jax_disable_jit', True)

# Define config file - could change to FLAGS at some point
config = {'seed': 123,
          'epochs': int(1e4),
          'early_stopping': jnp.array(1000),
          'batch_size': 256,
          'expectile': 0.5,
          'baseline_reward': 0,
          'interval_probability': 1.0,
          'top_actions_quantile': 0.75,
          'gamma': 0.99,
          'actor_lr': 0.001,
          'value_lr': 0.001,
          'critic_lr': 0.001,
          'hidden_dims': (256, 256),
          'clipping': 1,
          'top_bar_coord': 1.2,  # 0.9,
          'bottom_bar_coord': 0.8,  # 0.5
          }

if __name__ == "__main__":
    from core.agent import IQLAgent
    from core.common import (load_data, progress_bar, alter_batch, Batch, filter_to_action,
                             calc_traj_discounted_rewards, downsample_batch,
                             move_to_gpu, filter_dataset)
    from core.evaluate import evaluate_envs, run_and_animate
    from core.envs import make_variable_env

    # Set whether to train and/or evaluate
    logging_bool = False
    evaluate_bool = False

    if logging_bool:
        wandb.init(
            project="IQL-VariableTimeSteps-Formal",
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
        path=f'./offline_datasets/LunarLander/{interval_probability}_probability_5_steps/dataset_reward_{baseline_reward}.pkl',
        scale='standardise',
        gamma=config['gamma'])

    # Start by defining the intervals between actions (both the current and next action)
    # intervals = the actual number of steps between actions
    intervals = np.array([len(traj) for traj in loaded_data.rewards])
    intervals_unique = np.unique(intervals)
    mapping = {interval: idx for idx, interval in enumerate(intervals_unique)}
    len_actions = np.array([mapping[interval] for interval in intervals])
    next_len_actions = np.roll(len_actions, -1)

    intervals = jnp.array(intervals)
    len_actions = jnp.array(len_actions)
    next_len_actions = jnp.array(next_len_actions)

    # Calculate rewards
    rewards = calc_traj_discounted_rewards(loaded_data.rewards, config['gamma'])
    loaded_data = alter_batch(loaded_data, rewards=rewards, len_actions=len_actions, next_len_actions=next_len_actions,
                              intervals=intervals)
    del rewards

    # Move to GPU
    loaded_data = move_to_gpu(loaded_data, gpu_keys=['states', 'actions', 'discounted_rewards', 'episode_rewards',
                                                     'next_states', 'dones', 'action_logprobs', 'len_actions',
                                                     'rewards'])


    # Make sure this matches with the desired dataset's extra_step metadata
    def extra_step_filter(x):
        # If in rectangle
        if config['bottom_bar_coord'] < x[1] < config['top_bar_coord']:
            # Slow zone
            if np.random.uniform() < interval_probability:
                return 5
        # Otherwise, normal time steps (no delay)
        return 0


    # Train agent
    def train(data):
        # Create agent
        dummy_env = make_env('LunarLander-v2')

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

        print('\n\n', '=' * 50, '\n', ' ' * 3, '\U0001F483' * 3, ' ' * 1, f'Training networks',
              ' ' * 2, '\U0001F483' * 3, '\n', '=' * 50, '\n')

        best_loss = jnp.array(jnp.inf)
        total_training_steps = jnp.array(0)
        count = jnp.array(0)

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
                                       int(config[f'batch_size'] * (1 / (1 - config['top_actions_quantile']))))

            value_batch = downsample_batch(batch, jax.random.PRNGKey(123), config['batch_size'])
            # Use TD learning for the value and critic networks (based on the value network)
            next_state_values = agent.value(value_batch.next_states)[1]

            gammas = jnp.ones(shape=len(value_batch.rewards)) * config['gamma']
            gammas = jnp.power(gammas, value_batch.intervals)
            discounted_rewards = value_batch.rewards + gammas * next_state_values * (1 - value_batch.dones)

            value_batch = alter_batch(value_batch, discounted_rewards=discounted_rewards)
            del discounted_rewards

            # Use advantage for the actor network
            critic_values = jnp.minimum(*agent.critic(batch.states)[1])
            critic_values = filter_to_action(critic_values, batch.actions)

            state_values = agent.value(batch.states)[1]

            advantages = critic_values - state_values

            batch = alter_batch(batch, advantages=advantages)

            # Filter for top actions
            filter_point = jnp.quantile(advantages, config['top_actions_quantile'])

            batch = filter_dataset(batch, value_batch.advantages > filter_point,
                                         target_keys=['states', 'actions', 'advantages'])

            # Remove excess data from the batch
            batch = batch._asdict()
            value_batch = value_batch._asdict()
            for key in ['episode_rewards', 'next_states', 'next_actions', 'action_logprobs']:
                batch[key] = None
                value_batch[key] = None
            batch = Batch(**batch)
            value_batch = Batch(**value_batch)

            # Perform the update step
            value_loss_info = agent.update_async(value_batch,
                                                 value_loss_fn={'expectile': 0},
                                                 critic_loss_fn={'mc_mse': 0},
                                                 expectile=config['expectile'],
                                                 critic=True,
                                                 value=True,
                                                 )

            loss_info = agent.update_async(batch,
                                           actor_loss_fn={'clone': 0},
                                           actor=True
                                           )

            loss_info.update(value_loss_info)

            # Log intermittently
            if epoch % 5 == 0:

                if logging_bool:
                    # Log results
                    logged_results = {'training_step': total_training_steps,
                                      'gradient_step': epoch,
                                      f'{current_net}_loss': loss_info[loss_key]}
                    wandb.log(logged_results)

        # Save each model at the end of training
        agent.actor.save(os.path.join(model_dir, 'model_checkpoints/actor')) if is_net(
            'actor') else None
        agent.critic.save(os.path.join(model_dir, 'model_checkpoints/critic')) if is_net(
            'critic') else None
        agent.value.save(os.path.join(model_dir, 'model_checkpoints/value')) if is_net(
            'value') else None

        # Evaluate agent
        n_envs = 1000
        print('\n\n', '=' * 50, '\n', ' ' * 3, '\U0001F514' * 3, ' ' * 1, f'Evaluating network', ' ' * 2,
              '\U0001F514' * 3, '\n', '=' * 50)
        episode_rewards = evaluate_envs(agent, make_vec_env(lambda: make_variable_env('LunarLander-v2',
                                                                                      fn=extra_step_filter),
                                                            n_envs=n_envs))
        print(f'\nMedian reward: {np.median(episode_rewards)}')
        print(f'Mean reward: {np.mean(episode_rewards)}')

        with open(os.path.join(model_dir, 'rewards.txt'), 'w') as f:
            f.write(f'Baseline reward: {baseline_reward}\n')
            f.write(f'Median reward: {np.median(episode_rewards)}\n')
            f.write(f'Mean reward: {np.mean(episode_rewards)}\n')
            f.close()

        if logging_bool:
            wandb.define_metric('median_reward', summary='max')
            wandb.define_metric('mean_reward', summary='max')

            wandb.log({'median_reward': np.median(episode_rewards)})
            wandb.log({'mean_reward': np.mean(episode_rewards)})

        return agent


    # Run the train script
    agent = train(loaded_data)

    # ============================================================== #
    # ======================== EVALUATION ========================== #
    # ============================================================== #

    if evaluate_bool:
        dummy_env = make_env('LunarLander-v2')

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
        results = evaluate_envs(agent, make_vec_env(lambda: make_variable_env('LunarLander-v2',
                                                                              fn=extra_step_filter),
                                                    n_envs=envs_to_evaluate))
        print(f'\nMedian reward: {np.median(results)}')
        print(f'\nMean reward: {np.mean(results)}')

        # Animate the agent's performance
        print('\n\n', '=' * 50, '\n', ' ' * 3, '\U0001F4FA' * 3, ' ' * 1, f'Generating gifs', ' ' * 2,
              '\U0001F4FA' * 3, '\n', '=' * 50)
        env = make_variable_env('LunarLander-v2', fn=extra_step_filter, render_mode='rgb_array')
        run_and_animate(agent, env, runs=20, directory=os.path.join(model_dir, 'gifs'), **config)
