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
          'epochs': int(1e4),
          'steps': None,
          'batch_size': int(1e5),
          'n_envs': 20,
          'expectile': 0.5,
          'gamma': 0.99,
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

    del dummy_env

    # Load static dataset
    print('Loading and processing dataset...')
    data = load_data(path='./offline_datasets/LunarLander/dataset_1/dataset_reward_1.pkl',
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

        for value, critic, actor in [[True, False, False], [False, True, False], [False, False, True]]:

            loss_key = f"{'value' if value else ('critic' if critic else 'actor')}_loss"
            best_loss = jnp.inf
            total_training_steps = 0
            count = 0

            for epoch in range(config['epochs']):
                if epoch > 0 and epoch % 100 == 0:
                    print(f'\n\n{epoch} epochs complete!\n\n')
                progress_bar(epoch % 100, 100)
                batch = agent.sample(data,
                                     config['batch_size'])

                # Calculate next state values, discounted rewards (for critic update), and current advantages
                next_state_values = agent.value(batch.next_states)[1] if critic else None
                rewards = calc_traj_discounted_rewards(batch.rewards, config['gamma'])

                advantages = filter_to_action(jnp.minimum(*agent.critic(batch.states)[1]),
                                              batch.actions) - agent.value(batch.states)[1] if actor else None
                batch = alter_batch(batch, advantages=advantages)

                # Remove excess from the batch
                excess_data = {}
                batch = batch._asdict()
                for key in ['episode_rewards', 'next_states', 'next_actions', 'action_logprobs']:
                    excess_data[key] = batch[key]
                    batch[key] = None
                batch = Batch(**batch)

                loss_info = agent.update_async(batch,
                                               actor,
                                               critic,
                                               value,
                                               value_loss_fn={'expectile': 0},
                                               critic_loss_fn={'td_mse': 0},
                                               actor_loss_fn={'iql': 0},
                                               expectile=config['expectile'],
                                               gamma=config['gamma'],
                                               rewards=rewards,
                                               next_state_values=next_state_values,)

                total_training_steps += config['batch_size']

                # Record best loss
                if loss_info[loss_key] < best_loss:
                    best_loss = loss_info[loss_key]
                    count = 0
                    agent.actor.save(os.path.join('./experiments', agent.path, 'actor')) if actor else None
                    agent.critic.save(os.path.join('./experiments', agent.path, 'critic')) if critic else None
                    agent.value.save(os.path.join('./experiments', agent.path, 'value')) if value else None
                else:
                    count += 1
                    if count > 1000:
                        agent.actor = agent.actor.load(
                            os.path.join('./experiments', agent.path, 'actor')) if actor else agent.actor
                        agent.critic = agent.critic.load(
                            os.path.join('./experiments', agent.path, 'critic')) if critic else agent.critic
                        agent.value = agent.value.load(
                            os.path.join('./experiments', agent.path, 'value')) if value else agent.value
                        break

                # Log intermittently
                if logging:
                    # Log results
                    logged_results = {}
                    for key, trigger in [['actor', actor], ['value', value], ['critic', critic]]:
                        if trigger:
                            logged_results[f'{key}_loss'] = loss_info[f'{key}_loss']
                    wandb.log(logged_results, step=total_training_steps)

    # ============================================================== #
    # ======================== EVALUATION ========================== #
    # ============================================================== #

    if evaluate:
        filename = agent.path
        agent.actor = agent.actor.load(os.path.join('../experiments', f'{filename}', 'actor'))
        agent.critic = agent.critic.load(os.path.join('../experiments', f'{filename}', 'critic'))
        agent.value = agent.value.load(os.path.join('../experiments', f'{filename}', 'value'))

        max_episode_steps = 1000
        envs_to_evaluate = 1000

        results = evaluate_envs(agent, make_vec_env('LunarLander-v2', n_envs=envs_to_evaluate))
        print(f'\nMedian reward: {np.median(results)}')

        # Animate the agent's performance
        env = make_variable_env('LunarLander-v2', fn=extra_step_filter, render_mode='rgb_array')
        run_and_animate(agent, env, runs=20, directory=os.path.join(agent.path, 'gifs'), **config)

