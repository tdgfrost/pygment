import numpy as np
from gymnasium.envs import make as make_env
import os
import jax.numpy as jnp
from stable_baselines3.common.env_util import make_vec_env
import wandb
from math import ceil
import jax

# Set jax to CPU
# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)
# jax.config.update('jax_disable_jit', True)

# Define config file - could change to FLAGS at some point
config = {'seed': 123,
          'epochs': 100000,
          'env_id': 'LunarLander-v2',
          'early_stopping': jnp.array(1000),
          'batch_size': 1024,
          'monte_carlo_sample_size': 20,
          'step_delay': 11,
          'sync_steps': 20,
          'expectile': 0.5,
          'baseline_reward': 45,
          'n_episodes': 10000,
          'interval_probability': 0.25,
          'top_actions_quantile': 0.5,
          'filter_point': 0,
          'gamma': 0.99,
          'lr': 0.001,
          'dropout_rate': 0.1,
          'alpha_soft_update': 1,
          'hidden_dims': (256, 256),
          'clipping': 1,
          'top_bar_coord': 1.2,  # 0.9,
          'bottom_bar_coord': 0.8,  # 0.5
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
        # If in rectangle
        if config['bottom_bar_coord'] < x[1] < config['top_bar_coord']:
            # Slow zone
            if np.random.uniform() < config['interval_probability']:
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

            # Learn the Q values
            agent.refresh_keys()
            next_state_values = agent.target_value(jnp.tile(batch.next_states, (config['monte_carlo_sample_size'],
                                                                                1, 1)),
                                                   rngs={'dropout': agent.target_value_key})[1]

            discounted_rewards_for_critic = (np.array(batch.rewards)
                                             + gammas * next_state_values.mean(0) * (1 - np.array(batch.dones)))

            batch = alter_batch(batch, discounted_rewards=jnp.array(discounted_rewards_for_critic), next_states=None,
                                dones=None, intervals=None, rewards=None)

            # Perform the update step for critic networks
            critic_loss_info = agent.update_async(batch,
                                                  critic_loss_fn={'mse': 0},
                                                  critic=True)

            # Learn the expectile V(s) values
            agent.refresh_keys()
            q1, q2 = agent.critic(jnp.tile(batch.states, (config['monte_carlo_sample_size'], 1, 1)),
                                  rngs={'dropout': agent.critic_key})[1]
            discounted_rewards_for_value = jnp.minimum(q1.mean(0), q2.mean(0))
            discounted_rewards_for_value = filter_to_action(discounted_rewards_for_value, batch.actions)

            batch = alter_batch(batch, discounted_rewards=discounted_rewards_for_value)

            # Perform the update step for interval value and critic networks
            value_loss_info = agent.update_async(batch,
                                                 value_loss_fn={'expectile': 0},
                                                 expectile=config['expectile'],
                                                 value=True)

            value_loss_info.update(critic_loss_info)

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

            if epoch % 100 == 0:
                # Save each model
                agent.target_value.save(os.path.join(model_dir, 'model_checkpoints/target_value'))
                agent.critic.save(os.path.join(model_dir, 'model_checkpoints/critic'))
                agent.value.save(os.path.join(model_dir, 'model_checkpoints/value'))

        # And train the actor
        total_training_steps = jnp.array(0)

        # Calculate the advantages
        # Perform MC dropout inference to assess uncertainty in values
        print('\n\n', '=' * 50, '\n', ' ' * 3, '\U0001F9D9' * 3, ' ' * 1, f'Calculating uncertainty',
              ' ' * 2, '\U0001F9D9' * 3, '\n', '=' * 50, '\n')

        step_size = int(5e5) // config['monte_carlo_sample_size']

        def iter_through_data(input_states, actions, current_agent, mc_sample_size=config['monte_carlo_sample_size']):
            def reduce(x):
                return np.array([jnp.mean(x, -1), jnp.var(x, -1)]).T

            def reduce_critic(actions, q_tuple): # input needs to be actions[idx].reshape(-1)
                q_s = []
                for q in q_tuple:
                    q_s += [reduce(filter_to_action(q.reshape(-1, current_agent.action_dim),
                                            actions).reshape(-1, mc_sample_size))]
                return q_s[0], q_s[1]

            current_sample_values = []
            current_critic_values_1 = []
            current_critic_values_2 = []

            for i in range(ceil(input_states.shape[0] / step_size)):
                progress_bar(i, ceil(input_states.shape[0] / step_size))
                current_agent.refresh_keys()
                idx = slice(i * step_size, (i + 1) * step_size, 1)
                # Remove the resizes where it isn't needed
                current_sample_values += [reduce(
                    current_agent.target_value(input_states[idx],
                                               rngs={'dropout': current_agent.target_value_key}
                                               )[1].reshape(-1, mc_sample_size))]

                (critic_value_iter_1,
                 critic_value_iter_2) = reduce_critic(actions[idx].reshape(-1),
                                                      current_agent.critic(input_states[idx],
                                                                           rngs={'dropout':
                                                                                     current_agent.critic_key})[1])

                current_critic_values_1 += [critic_value_iter_1]
                current_critic_values_2 += [critic_value_iter_2]

            current_sample_values = np.concatenate(current_sample_values)
            current_critic_values_1 = np.concatenate(current_critic_values_1)
            current_critic_values_2 = np.concatenate(current_critic_values_2)

            return (current_agent,
                    (current_sample_values[:, 0], current_sample_values[:, 1]),
                    (current_critic_values_1[:, 0], current_critic_values_1[:, 1]),
                    (current_critic_values_2[:, 0], current_critic_values_2[:, 1]))

        (agent,
         (state_values_mu, state_values_var),
         (q1_mu, q1_var),
         (q2_mu, q2_var)) = iter_through_data(
            jnp.tile(jnp.expand_dims(data.states, 1),
                     (1, config['monte_carlo_sample_size'], 1)),
            jnp.tile(jnp.expand_dims(data.actions, 1),
                     (1, config['monte_carlo_sample_size'])),
            agent)

        q_bool = q1_mu < q2_mu

        critic_values_mu, critic_values_var = np.where(q_bool, q1_mu, q2_mu), np.where(q_bool, q1_var, q2_var)

        advantages = critic_values_mu - state_values_mu

        """
        advantages_std = np.sqrt(state_values_std ** 2 + critic_values_std ** 2)
        cdf = 1 - norm.cdf(0, advantages, advantages_std)
        if 'gaussian_confidence' in config.keys() and config['gaussian_confidence'] is not None:
            filter_point = config['gaussian_confidence']
        else:
            filter_point = np.quantile(cdf, config['confidence_quantile'])
        """

        # Filter for top half of actions
        if 'filter_point' not in config.keys():
            filter_point = np.quantile(advantages, config['top_actions_quantile'])
        else:
            filter_point = config['filter_point']

        if np.sum(advantages > filter_point) < config['batch_size']:
            return agent

        data = filter_dataset(data, advantages > filter_point,
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
