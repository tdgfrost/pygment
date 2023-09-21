import jax
import numpy as np
from gymnasium.envs import make as make_env
import os
import jax.numpy as jnp
from stable_baselines3.common.env_util import make_vec_env
import wandb

# Set jax to CPU
# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)
# jax.config.update('jax_disable_jit', True)

# Define config file - could change to FLAGS at some point
config = {'seed': 123,
          'epochs': int(1e6),
          'early_stopping': 200,
          'value_batch_size': 1024,
          'critic_batch_size': 1024,
          'actor_batch_size': 1024 / (1-0.9),
          # Need to separate out batch sizes for value/critic networks (all data) and actor (only the subsampled data)
          'expectile': 0.9,
          'gamma': 0.999,
          'actor_lr': 0.001,
          'value_lr': 0.001,
          'critic_lr': 0.001,
          'hidden_dims': (64, 64),
          'clipping': 1,
          'top_bar_coord': 1.2,  # 0.9,
          'bottom_bar_coord': 0.8,  # 0.5
          }

if __name__ == "__main__":
    from core.agent import IQLAgent
    from core.common import load_data, progress_bar, alter_batch, Batch, filter_to_action, calc_traj_discounted_rewards
    from core.evaluate import evaluate_envs, run_and_animate
    from core.envs import make_variable_env

    # Set whether to train and/or evaluate
    train = True
    logging = True
    evaluate = True

    # Create agent
    dummy_env = make_env('LunarLander-v2')

    agent = IQLAgent(observations=dummy_env.observation_space.sample(),
                     action_dim=dummy_env.action_space.n,
                     dropout_rate=None,
                     opt_decay_schedule="cosine",
                     **config)

    model_dir = os.path.join('./experiments/IQL', agent.path)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'config.txt'), 'w') as f:
        f.write(str(config))
        f.close()

    del dummy_env

    # Load static dataset
    print('Loading and processing dataset...')
    data = load_data(path='./offline_datasets/LunarLander/dataset_4/dataset_combined.pkl',
                     scale='standardise',
                     gamma=config['gamma'])

    # Make sure this matches with the desired dataset's extra_step metadata
    def extra_step_filter(x):
        # If in rectangle
        if config['bottom_bar_coord'] < x[1] < config['top_bar_coord']:
            # with p == 0.05, delay by 20 steps
            if np.random.uniform() < 0.05:
                return 20
        # Otherwise, normal time steps (no delay)
        return 0


    # ============================================================== #
    # ========================= TRAINING =========================== #
    # ============================================================== #

    # Train agent
    if train:
        if logging:
            os.environ['WANDB_BASE_URL'] = "http://localhost:8080"
            # Prepare logging
            wandb.init(
                project="IQL-VariableTimeSteps",
                config=config,
            )

        for current_net in ['value', 'critic', 'actor']:
            print('\n\n', '=' * 50, '\n', ' ' * 3, '\U0001F483' * 3, ' ' * 1, f'Training {current_net} network', ' ' * 2,
                  '\U0001F483' * 3, '\n', '=' * 50, '\n')
            # loss_key = f"{'value' if value else ('critic' if critic else 'actor')}_loss"
            loss_key = f'{current_net}_loss'
            best_loss = jnp.inf
            total_training_steps = 0
            count = 0
            def is_net(x): return x == current_net

            for epoch in range(config['epochs']):
                if epoch > 0 and epoch % 100 == 0:
                    print(f'\n\n{epoch} epochs complete!\n')
                progress_bar(epoch % 100, 100)
                batch = agent.sample(data,
                                     config[f'{current_net}_batch_size'])

                # Calculate next state values, discounted rewards (for critic update), and current advantages
                next_state_values = agent.value(batch.next_states)[1] if is_net('critic') else None
                rewards = calc_traj_discounted_rewards(batch.rewards, config['gamma'])

                advantages = (filter_to_action(jnp.minimum(*agent.critic(batch.states)[1]),
                                               batch.actions)
                              - agent.value(batch.states)[1]) if is_net('actor') else None
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
                                               critic_loss_fn={'td_mse': 0},
                                               actor_loss_fn={'iql': 0},
                                               expectile=config['expectile'],
                                               gamma=config['gamma'],
                                               rewards=rewards,
                                               next_state_values=next_state_values,
                                               **{current_net: True})

                total_training_steps += config[f'{current_net}_batch_size']

                # Record best loss
                if loss_info[loss_key] < best_loss:
                    best_loss = loss_info[loss_key]
                    count = 0
                    agent.actor.save(os.path.join(model_dir, 'model_checkpoints/actor')) if is_net('actor') else None
                    agent.critic.save(os.path.join(model_dir, 'model_checkpoints/critic')) if is_net('critic') else None
                    agent.value.save(os.path.join(model_dir, 'model_checkpoints/value')) if is_net('value') else None
                else:
                    count += 1
                    if count > config['early_stopping']:
                        agent.actor = agent.actor.load(
                            os.path.join(model_dir, 'model_checkpoints/actor')) if is_net('actor') else agent.actor
                        agent.critic = agent.critic.load(
                            os.path.join(model_dir, 'model_checkpoints/critic')) if is_net('critic') else agent.critic
                        agent.value = agent.value.load(
                            os.path.join(model_dir, 'model_checkpoints/value')) if is_net('value') else agent.value
                        break

                # Log intermittently
                if logging:
                    # Log results
                    logged_results = {'training_step': total_training_steps,
                                      'gradient_step': epoch,
                                      f'{current_net}_loss': loss_info[f'{current_net}_loss']}
                    wandb.log(logged_results)

    # ============================================================== #
    # ======================== EVALUATION ========================== #
    # ============================================================== #

    if evaluate:
        filename = agent.path
        agent.actor = agent.actor.load(os.path.join(model_dir, 'model_checkpoints/actor'))

        max_episode_steps = 1000
        envs_to_evaluate = 1000

        print('\n\n', '=' * 50, '\n', ' ' * 3, '\U0001F514' * 3, ' ' * 1, f'Evaluating network', ' ' * 2,
              '\U0001F514' * 3, '\n', '=' * 50)
        results = evaluate_envs(agent, make_vec_env('LunarLander-v2', n_envs=envs_to_evaluate))
        print(f'\nMedian reward: {np.median(results)}')

        # Animate the agent's performance
        print('\n\n', '=' * 50, '\n', ' ' * 3, '\U0001F4FA' * 3, ' ' * 1, f'Generating gifs', ' ' * 2,
              '\U0001F4FA' * 3, '\n', '=' * 50)
        env = make_variable_env('LunarLander-v2', fn=extra_step_filter, render_mode='rgb_array')
        run_and_animate(agent, env, runs=20, directory=os.path.join(model_dir, 'gifs'), **config)
