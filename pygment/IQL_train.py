import jax
import numpy as np
from gymnasium.envs import make as make_env
import os
import jax.numpy as jnp
from stable_baselines3.common.env_util import make_vec_env
import wandb
import flax.linen as nn
import yaml

# Set jax to CPU
# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)
# jax.config.update('jax_disable_jit', True)

# Define config file - could change to FLAGS at some point
config = {'seed': 123,
          'epochs': int(1e4),
          'early_stopping': 1000,
          'value_batch_size': 256,
          'critic_batch_size': 256,
          'interval_batch_size': 256,
          'actor_batch_size': 256,
          # Need to separate out batch sizes for value/critic networks (all data) and actor (only the subsampled data)
          'expectile': 0.5,
          'gamma': 0.99,
          'actor_lr': 0.001,
          'value_lr': 0.001,
          'critic_lr': 0.001,
          'interval_lr': 0.001,
          'hidden_dims': (64, 64),
          'clipping': 1,
          'top_bar_coord': 1.2,  # 0.9,
          'bottom_bar_coord': 0.8,  # 0.5
          }

if __name__ == "__main__":
    from core.agent import IQLAgent
    from core.common import (load_data, progress_bar, alter_batch, Batch, filter_to_action, calc_traj_discounted_rewards,
                             move_to_gpu)
    from core.evaluate import evaluate_envs, run_and_animate
    from core.envs import make_variable_env

    # Set whether to train and/or evaluate
    logging_bool = False
    evaluate_bool = False

    # Update actor batch size to match expectile
    config['actor_batch_size'] = int(config['actor_batch_size'] / (1 - config['expectile']))

    # ============================================================== #
    # ========================= TRAINING =========================== #
    # ============================================================== #

    # Load static dataset
    print('Loading and processing dataset...')
    baseline_reward = 4
    data = load_data(path=f'./offline_datasets/LunarLander/dataset_reward_{baseline_reward}.pkl',
                     scale='standardise',
                     gamma=config['gamma'])

    intervals = jnp.array([len(traj) for traj in data.rewards])
    interval_range = intervals.max() - intervals.min() + 1  # Range is inclusive so add 1 to this number

    # Calculate len_actions and state_masks (for later, before moving to GPU)
    len_actions = jnp.array([len(traj) for traj in data.rewards]) - intervals.min()
    rewards = calc_traj_discounted_rewards(data.rewards, config['gamma'])
    data = alter_batch(data, rewards=rewards, len_actions=len_actions)
    del rewards

    state_masks = jnp.zeros(shape=(len(data.len_actions), interval_range),
                            dtype=jnp.bool_)
    idxs = jnp.column_stack((jnp.arange(len(data.len_actions)), data.len_actions))
    state_masks = state_masks.at[idxs[:, 0], idxs[:, 1]].set(True)

    # Move to GPU
    data = move_to_gpu(data, gpu_keys=['states', 'actions', 'discounted_rewards', 'episode_rewards',
                                       'next_states', 'dones', 'action_logprobs', 'len_actions', 'rewards'])

    # Make sure this matches with the desired dataset's extra_step metadata
    def extra_step_filter(x):
        # If in rectangle
        if config['top_bar_coord'] < x[1] < config['bottom_bar_coord']:
            # with p == 0.05, delay by 20 steps
            if np.random.uniform() < 0.05:
                return 20
        # Otherwise, normal time steps (no delay)
        return 0

    # Train agent
    def train():
        if logging_bool:
            wandb.init(
                project="IQL-VariableTimeSteps",
                config=config,
            )

        # Create agent
        dummy_env = make_env('LunarLander-v2')

        agent = IQLAgent(observations=dummy_env.observation_space.sample(),
                         action_dim=dummy_env.action_space.n,
                         interval_dim=interval_range.item(),
                         interval_min=intervals.min().item(),
                         opt_decay_schedule="cosine",
                         **config)

        model_dir = os.path.join('./experiments/IQL', agent.path)
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'config.txt'), 'w') as f:
            f.write(str(config))
            f.close()

        del dummy_env

        # Standardise the state inputs
        agent.standardise_inputs(data.states)

        for current_net in ['interval', 'value', 'critic', 'actor']:
            print('\n\n', '=' * 50, '\n', ' ' * 3, '\U0001F483' * 3, ' ' * 1, f'Training {current_net} network',
                  ' ' * 2,
                  '\U0001F483' * 3, '\n', '=' * 50, '\n')
            # loss_key = f"{'value' if value else ('critic' if critic else 'actor')}_loss"
            loss_key = f'{current_net}_loss'
            best_loss = jnp.inf
            total_training_steps = 0
            count = 0

            if logging_bool:
                # Keep track of the best loss values
                wandb.define_metric(f'{current_net}_loss', summary='min')

            def is_net(x):
                return x == current_net

            for epoch in range(config['epochs']):
                if epoch > 0 and epoch % 100 == 0:
                    print(f'\n\n{epoch} epochs complete!\n')
                progress_bar(epoch % 100, 100)
                batch, idxs = agent.sample(data,
                                           config[f'{current_net}_batch_size'])

                # Calculate next state values
                next_state_values = None
                if is_net('critic'):
                    # Calculate the real TD value from next_state_value + current_state rewards
                    next_state_values = agent.value(batch.next_states)[1]
                    next_state_intervals = nn.softmax(agent.interval(batch.next_states)[1], axis=-1)
                    next_state_values = (next_state_values * next_state_intervals).sum(-1)

                    gammas = jnp.ones(shape=len(batch.rewards)) * config['gamma']
                    gammas = jnp.power(gammas, batch.len_actions + intervals.min())
                    rewards = batch.rewards + gammas * next_state_values * (1 - batch.dones)
                    batch = alter_batch(batch, rewards=rewards)
                    del rewards

                    # Then calculate the rest of the theoretical current state values,
                    # and insert the real TD value at the appropriate place
                    current_state_values = agent.value(batch.states)[1]
                    current_state_mask = state_masks[idxs]
                    """
                    current_state_mask = jnp.array(
                        [[False if bool_idx != action_len else True for bool_idx in range(interval_range)]
                         for action_len in batch.len_actions])
                    """

                    current_state_values = current_state_values.at[current_state_mask].set(batch.rewards)

                    # Finally, multiply by the probability distribution and sum for an average
                    current_state_intervals = nn.softmax(agent.interval(batch.states)[1], axis=-1)
                    discounted_rewards = (current_state_intervals * current_state_values).sum(-1)

                    batch = alter_batch(batch, discounted_rewards=discounted_rewards)

                # Calculate advantages
                advantages = None
                if is_net('actor'):
                    critic_values = filter_to_action(jnp.minimum(*agent.critic(batch.states)[1]),
                                                     batch.actions)

                    state_values = agent.value(batch.states)[1]
                    state_intervals = nn.softmax(agent.interval(batch.states)[1], axis=-1)
                    state_values = (state_values * state_intervals).sum(-1)

                    advantages = critic_values - state_values

                    # Try normalising advantages
                    advantages = (advantages - advantages.mean()) / jnp.maximum(advantages.std(), 1e-8)

                batch = alter_batch(batch, advantages=advantages)

                # Remove excess from the batch
                excess_data = {}
                batch = batch._asdict()
                for key in ['episode_rewards', 'next_states', 'next_actions', 'action_logprobs']:
                    excess_data[key] = batch[key]
                    batch[key] = None
                batch = Batch(**batch)
                loss_info = agent.update_async(batch,
                                               value_loss_fn={'expectile': 0},
                                               # critic_loss_fn={'td_mse': 0},
                                               critic_loss_fn={'mc_mse': 0},
                                               actor_loss_fn={'iql': 0},
                                               expectile=config['expectile'],
                                               gamma=config['gamma'],
                                               next_state_values=next_state_values,
                                               **{current_net: True})

                total_training_steps += config[f'{current_net}_batch_size']
                """
                # Record best loss
                if loss_info[loss_key] < best_loss:
                    best_loss = loss_info[loss_key]
                    count = 0
                    '''
                    agent.actor.save(os.path.join(model_dir, 'model_checkpoints/actor')) if is_net('actor') else None
                    agent.critic.save(os.path.join(model_dir, 'model_checkpoints/critic')) if is_net('critic') else None
                    agent.value.save(os.path.join(model_dir, 'model_checkpoints/value')) if is_net('value') else None
                    agent.interval.save(os.path.join(model_dir, 'model_checkpoints/interval')) if is_net(
                        'interval') else None
                    '''
                else:
                    count += 1
                    if count > config['early_stopping']:
                        '''
                        agent.actor = agent.actor.load(
                            os.path.join(model_dir, 'model_checkpoints/actor')) if is_net('actor') else agent.actor
                        agent.critic = agent.critic.load(
                            os.path.join(model_dir, 'model_checkpoints/critic')) if is_net('critic') else agent.critic
                        agent.value = agent.value.load(
                            os.path.join(model_dir, 'model_checkpoints/value')) if is_net('value') else agent.value
                        agent.interval = agent.interval.load(
                            os.path.join(model_dir, 'model_checkpoints/interval')) if is_net('interval') else agent.interval
                        '''
                        agent.actor.save(os.path.join(model_dir, 'model_checkpoints/actor')) if is_net(
                            'actor') else None
                        agent.critic.save(os.path.join(model_dir, 'model_checkpoints/critic')) if is_net(
                            'critic') else None
                        agent.value.save(os.path.join(model_dir, 'model_checkpoints/value')) if is_net(
                            'value') else None
                        agent.interval.save(os.path.join(model_dir, 'model_checkpoints/interval')) if is_net(
                            'interval') else None
                        break

                # Log intermittently
                if logging_bool:
                    # Log results
                    logged_results = {'training_step': total_training_steps,
                                      'gradient_step': epoch,
                                      f'{current_net}_loss': loss_info[f'{current_net}_loss']}
                    wandb.log(logged_results)
                """
        # Evaluate agent
        n_envs = 1000
        print('\n\n', '=' * 50, '\n', ' ' * 3, '\U0001F514' * 3, ' ' * 1, f'Evaluating network', ' ' * 2,
              '\U0001F514' * 3, '\n', '=' * 50)
        episode_rewards = evaluate_envs(agent, make_vec_env(lambda: make_variable_env('LunarLander-v2',
                                                                                      fn=extra_step_filter),
                                                            n_envs=n_envs))
        print(f'\nMedian reward: {np.median(episode_rewards)}')

        with open('rewards.txt', 'w') as f:
            f.write(f'Baseline reward: {baseline_reward}\n')
            f.write(f'Median reward: {np.median(episode_rewards)}\n')
            f.close()

        if logging_bool:
            wandb.define_metric('median_reward', summary='max')

            wandb.log({'median_reward': np.median(episode_rewards)})

    # Run the train script
    train()

    # ============================================================== #
    # ======================== EVALUATION ========================== #
    # ============================================================== #

    if evaluate_bool:
        dummy_env = make_env('LunarLander-v2')

        agent = IQLAgent(observations=dummy_env.observation_space.sample(),
                         action_dim=dummy_env.action_space.n,
                         interval_dim=interval_range,
                         interval_min=intervals.min(),
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

        # Animate the agent's performance
        print('\n\n', '=' * 50, '\n', ' ' * 3, '\U0001F4FA' * 3, ' ' * 1, f'Generating gifs', ' ' * 2,
              '\U0001F4FA' * 3, '\n', '=' * 50)
        env = make_variable_env('LunarLander-v2', fn=extra_step_filter, render_mode='rgb_array')
        run_and_animate(agent, env, runs=20, directory=os.path.join(model_dir, 'gifs'), **config)
