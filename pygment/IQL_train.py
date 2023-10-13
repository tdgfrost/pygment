import jax
import numpy as np
from gymnasium.envs import make as make_env
import os
import jax.numpy as jnp
from stable_baselines3.common.env_util import make_vec_env
import wandb
import flax.linen as nn

# Set jax to CPU
jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)
# jax.config.update('jax_disable_jit', True)

# Define config file - could change to FLAGS at some point
config = {'seed': 123,
          'epochs': int(1e4),
          'early_stopping': jnp.array(1000),
          'value_batch_size': 256,
          'critic_batch_size': 256,
          'actor_batch_size': 256,
          'interval_batch_size': 256,
          'expectile': 0.75,
          'baseline_reward': 0,
          'interval_probability': 0.25,
          'top_actions_quantile': 0.5,
          'expectile_weighting': 0.5,
          'gamma': 0.99,
          'actor_lr': 0.001,
          'value_lr': 0.001,
          'critic_lr': 0.001,
          'interval_lr': 0.001,
          'hidden_dims': (256, 256),
          'clipping': 1,
          'top_bar_coord': 1.2,  # 0.9,
          'bottom_bar_coord': 0.8,  # 0.5
          }

if __name__ == "__main__":
    from core.agent import IQLAgent
    from core.common import (load_data, progress_bar, alter_batch, Batch, filter_to_action,
                             calc_traj_discounted_rewards, move_to_gpu, filter_dataset, move_params_to_gpu)
    from core.evaluate import evaluate_envs, run_and_animate
    from core.envs import make_variable_env

    # Set whether to train and/or evaluate
    logging_bool = False
    evaluate_bool = False
    training_device = jax.devices('cpu')[0]

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
                                                     'rewards'],
                              device=training_device)

    # Make sure this matches with the desired dataset's extra_step metadata
    def extra_step_filter(x):
        # If in rectangle
        if config['bottom_bar_coord'] < x[1] < config['top_bar_coord']:
            # Slow zone
            if np.random.uniform() < interval_probability:
                return 5
        # Otherwise, normal time steps (no delay)
        return 0


    def make_agent(path=None):
        dummy_env = make_env('LunarLander-v2')

        agent = IQLAgent(observations=dummy_env.observation_space.sample(),
                         action_dim=dummy_env.action_space.n,
                         intervals_unique=intervals_unique,
                         opt_decay_schedule="cosine",
                         **config)

        del dummy_env

        if path is None:
            agent_directory = os.path.join('./experiments/IQL', agent.path)
            agent.path = agent_directory
            os.makedirs(agent_directory, exist_ok=True)
            with open(os.path.join(agent_directory, 'config.txt'), 'w') as f:
                f.write(str(config))
                f.close()
        else:
            agent.path = path

        return agent.path, agent

    # Train agent
    def train(data):
        # Create agent
        model_dir, agent = make_agent()

        # Standardise the state inputs
        agent.standardise_inputs(data.states)

        # For advantage-prioritised cloning
        sample_prob = None

        for current_net in ['interval', 'value', 'critic', 'actor']:
            print('\n\n', '=' * 50, '\n', ' ' * 3, '\U0001F483' * 3, ' ' * 1, f'Training {current_net} network',
                  ' ' * 2, '\U0001F483' * 3, '\n', '=' * 50, '\n')

            def is_net(x):
                return x == current_net

            loss_key = f'{current_net}_loss'
            best_loss = jnp.array(jnp.inf)
            total_training_steps = jnp.array(0)
            count = jnp.array(0)

            if logging_bool:
                # Keep track of the best loss values
                wandb.define_metric(f'{current_net}_loss', summary='min')

            if is_net('actor'):
                print('Filtering dataset...')
                # We will perform the following using METAL for the faster inference across the whole dataset

                gpu_interval = agent.interval.change_device(jax.devices('METAL')[0])
                gpu_critic = agent.critic.change_device(jax.devices('METAL')[0])
                gpu_value = agent.value.change_device(jax.devices('METAL')[0])
                def metal(x): return jax.device_put(x, jax.devices('METAL')[0])
                def cpu(x): return jax.device_put(x, jax.devices('cpu')[0])

                # Calculate the interval probabilities for each state
                interval_values = nn.sigmoid(cpu(gpu_interval(metal(data.states))[1]))
                interval_values = jnp.hstack([1.0 - interval_values.reshape(-1, 1),
                                              interval_values.reshape(-1, 1)])

                # Calculate the critic and state values for each state
                # (using interval probabilities to marginalise the values)
                critic_values = cpu(jnp.minimum(*gpu_critic(metal(data.states))[1]))
                critic_values = filter_to_action(critic_values, data.actions)

                state_values = cpu(gpu_value(metal(data.states))[1])
                state_values = (state_values * interval_values).sum(-1)

                # Calculate the advantages
                advantages = critic_values - state_values
                advantages /= advantages.std()

                data = alter_batch(data, advantages=advantages)

                # Filter for top half of actions
                filter_point = np.quantile(np.array(advantages), config['top_actions_quantile'])

                print(np.array(advantages))

                data = filter_dataset(data, data.advantages > filter_point,
                                      target_keys=['states', 'actions', 'advantages'])

            for epoch in range(config['epochs']):
                if epoch > 0 and epoch % 100 == 0:
                    print(f'\n\n{epoch} epochs complete!\n')
                progress_bar(epoch % 100, 100)
                batch, idxs = agent.sample(data,
                                           config[f'{current_net}_batch_size'],
                                           p=sample_prob)

                # Use TD learning for the value and critic networks (based on the value network)
                if is_net('value') or is_net('critic'):
                    next_state_values = agent.value(batch.next_states)[1]
                    next_interval_values = nn.sigmoid(agent.interval(batch.next_states)[1])
                    next_interval_values = jnp.hstack([1.0 - next_interval_values.reshape(-1, 1),
                                                       next_interval_values.reshape(-1, 1)])
                    next_state_values = (next_state_values * next_interval_values).sum(-1)

                    gammas = jnp.ones(shape=len(batch.rewards)) * config['gamma']
                    gammas = jnp.power(gammas, batch.intervals)
                    discounted_rewards = batch.rewards + gammas * next_state_values * (1 - batch.dones)

                    batch = alter_batch(batch, discounted_rewards=discounted_rewards)
                    del discounted_rewards

                # Remove excess data from the batch
                excess_data = {}
                batch = batch._asdict()
                for key in ['episode_rewards', 'next_states', 'next_actions', 'action_logprobs']:
                    excess_data[key] = batch[key]
                    batch[key] = None
                batch = Batch(**batch)

                # Perform the update step
                loss_info = agent.update_async(batch,
                                               interval_loss_fn={'binary_crossentropy': 0},
                                               value_loss_fn={'expectile': 0},
                                               critic_loss_fn={'mc_mse': 0},
                                               actor_loss_fn={'iql': 0},
                                               expectile=config['expectile'],
                                               temperature=config['expectile_weighting'],
                                               **{current_net: True})

                total_training_steps += config[f'{current_net}_batch_size']

                # Record best loss
                if loss_info[loss_key] < best_loss:
                    best_loss = loss_info[loss_key]
                    count = jnp.array(0)

                else:
                    count += jnp.array(1)
                    if count > config['early_stopping'] and not is_net('value'):
                        break

                # Log intermittently
                if logging_bool:
                    # Log results
                    logged_results = {'training_step': total_training_steps,
                                      'gradient_step': epoch,
                                      f'{current_net}_loss': loss_info[loss_key]}
                    wandb.log(logged_results)

            # Save each model at the end of training
            agent.actor.save(os.path.join(model_dir, 'model_checkpoints/actor')) if is_net(
                'actor') else None
            agent.interval.save(os.path.join(model_dir, 'model_checkpoints/interval')) if is_net(
                'interval') else None
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
