import jax
import numpy as np
from gymnasium.envs import make as make_env
import os
import jax.numpy as jnp
from stable_baselines3.common.env_util import make_vec_env
import wandb
import flax.linen as nn
import pickle
import yaml

# Set jax to CPU
jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)
# jax.config.update('jax_disable_jit', True)

# Define config file - could change to FLAGS at some point
config = {'seed': 123,
          'epochs': int(3e4),
          'early_stopping': jnp.array(1000),
          'value_batch_size': 256,
          'critic_batch_size': 256,
          'actor_batch_size': 256,
          'interval_batch_size': 256,
          # Need to separate out batch sizes for value/critic networks (all data) and actor (only the subsampled data)
          'expectile': 0.85,
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
    from core.common import (load_data, progress_bar, alter_batch, Batch, filter_to_action, calc_traj_discounted_rewards,
                             move_to_gpu, filter_dataset)
    from core.evaluate import evaluate_envs, run_and_animate
    from core.envs import make_variable_env

    # Set whether to train and/or evaluate
    logging_bool = True
    evaluate_bool = False

    if logging_bool:
        wandb.init(
            project="IQL-VariableTimeSteps",
            config=config,
        )

    # Update actor batch size to match expectile
    # config['actor_batch_size'] = int(config['actor_batch_size'] / (1 - config['expectile']))

    # ============================================================== #
    # ========================= TRAINING =========================== #
    # ============================================================== #

    # Load static dataset
    print('Loading and processing dataset...')
    baseline_reward = 0
    loaded_data = load_data(path=f'./offline_datasets/LunarLander/1.0_probability_5_steps/dataset_reward_{baseline_reward}.pkl',
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
                                       'next_states', 'dones', 'action_logprobs', 'len_actions', 'rewards'])

    # Make sure this matches with the desired dataset's extra_step metadata
    def extra_step_filter(x):
        # If in rectangle
        if config['bottom_bar_coord'] < x[1] < config['top_bar_coord']:
            # with p == 0.05, delay by 20 steps
            # if np.random.uniform() < 0.05:
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
                """
                critic_filtered_idx = np.ravel_multi_index(np.array(jnp.vstack((data.actions, data.len_actions))),
                                                           (agent.action_dim, len(agent.intervals_unique)))
                critic_values = filter_to_action(jnp.minimum(*agent.critic(data.states)[1]),
                                                 critic_filtered_idx)
                """
                critic_values = jnp.minimum(*agent.critic(data.states)[1]).reshape(-1, agent.action_dim,
                                                                                   len(agent.intervals_unique))
                interval_values = nn.softmax(agent.interval(data.states)[1], -1).reshape(-1, 1,
                                                                                         len(agent.intervals_unique))
                critic_values = (critic_values * interval_values).sum(-1)
                critic_values = filter_to_action(critic_values, data.actions)

                state_values = agent.value(data.states)[1]
                #state_values = filter_to_action(state_values, data.len_actions)
                state_values = (state_values * interval_values.reshape(-1, len(agent.intervals_unique))).sum(-1)

                advantages = critic_values - state_values

                # Try normalising advantages
                advantages = (advantages - advantages.mean()) / advantages.std()

                data = alter_batch(data, advantages=advantages)

                data = filter_dataset(data, data.advantages > 0, target_keys=['states', 'actions', 'advantages'])

                # sample_prob = np.array(data.advantages, dtype=np.float64)
                # sample_prob = (sample_prob - sample_prob.mean()) / sample_prob.std()
                # sample_prob = np.exp(sample_prob) / np.exp(sample_prob).sum()
                # data = alter_batch(data, advantages=advantages)

            for epoch in range(config['epochs']):
                if epoch > 0 and epoch % 100 == 0:
                    print(f'\n\n{epoch} epochs complete!\n')
                progress_bar(epoch % 100, 100)
                batch, idxs = agent.sample(data,
                                           config[f'{current_net}_batch_size'],
                                           p=sample_prob)

                if is_net('value'):
                    # Calculate next state values
                    next_state_values = agent.value(batch.next_states)[1]
                    next_interval_values = nn.softmax(agent.interval(batch.next_states)[1], -1)
                    next_state_values = (next_state_values * next_interval_values).sum(-1)

                    gammas = jnp.ones(shape=len(batch.rewards)) * config['gamma']
                    gammas = jnp.power(gammas, batch.intervals)
                    discounted_rewards = batch.rewards + gammas * next_state_values * (1 - batch.dones)

                    batch = alter_batch(batch, discounted_rewards=discounted_rewards)
                    del discounted_rewards

                # Calculate next state values
                if is_net('critic'):
                    # Calculate the real TD value from next_state_value + current_state rewards
                    # next_state_values = agent.average_value(batch.next_states)[1]
                    next_state_values = agent.value(batch.next_states)[1]
                    # next_state_values = filter_to_action(next_state_values, batch.next_len_actions)
                    next_interval_values = nn.softmax(agent.interval(batch.next_states)[1], -1)
                    next_state_values = (next_state_values * next_interval_values).sum(-1)

                    gammas = jnp.ones(shape=len(batch.rewards)) * config['gamma']
                    gammas = jnp.power(gammas, batch.intervals)
                    discounted_rewards = batch.rewards + gammas * next_state_values * (1 - batch.dones)

                    actions = jnp.ravel_multi_index(jnp.vstack((batch.actions, batch.len_actions)),
                                                    (agent.action_dim, len(agent.intervals_unique)))

                    batch = alter_batch(batch, discounted_rewards=discounted_rewards, actions=actions)
                    del discounted_rewards

                # Remove excess from the batch
                excess_data = {}
                batch = batch._asdict()
                for key in ['episode_rewards', 'next_states', 'next_actions', 'action_logprobs']:
                    excess_data[key] = batch[key]
                    batch[key] = None
                batch = Batch(**batch)

                loss_info = agent.update_async(batch,
                                               interval_loss_fn={'crossentropy': 0},
                                               value_loss_fn={'expectile': 0},
                                               # value_loss_fn={'mc_mse': 0},
                                               critic_loss_fn={'mc_mse': 0},
                                               # critic_loss_fn={'expectile': 0},
                                               actor_loss_fn={'clone': 0},
                                               # actor_loss_fn={'iql': 0},
                                               expectile=config['expectile'],
                                               **{current_net: True})

                total_training_steps += config[f'{current_net}_batch_size']

                # Record best loss
                if loss_info[loss_key] < best_loss:
                    best_loss = loss_info[loss_key]
                    count = jnp.array(0)
                    '''
                    agent.actor.save(os.path.join(model_dir, 'model_checkpoints/actor')) if is_net('actor') else None
                    agent.critic.save(os.path.join(model_dir, 'model_checkpoints/critic')) if is_net('critic') else None
                    agent.value.save(os.path.join(model_dir, 'model_checkpoints/value')) if is_net('value') else None
                    '''
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
