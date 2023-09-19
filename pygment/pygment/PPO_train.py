import os
from gymnasium.envs import make as make_env
import jax
import numpy as np
import wandb
from tqdm import tqdm
from stable_baselines3.common.env_util import make_vec_env

# Set jax to CPU
# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)
# jax.config.update('jax_disable_jit', True)

# Define config file - could change to FLAGS at some point
config = {'seed': 123,
          'epochs': int(1e6),
          'steps': None,
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
    from core.agent import PPOAgent
    from core.common import progress_bar, shuffle_split_batch, alter_batch, flatten_batch
    from core.evaluate import evaluate_envs, run_and_animate
    from core.envs import EpisodeGenerator, make_variable_env

    # ============================================================== #
    # ======================== PREPARATION ========================= #
    # ============================================================== #

    # Set whether to train and/or evaluate
    train = True
    logging = True
    evaluate = True

    # Create agent
    dummy_env = make_env('LunarLander-v2')
    agent = PPOAgent(observations=dummy_env.observation_space.sample(),
                     action_dim=dummy_env.action_space.n,
                     opt_decay_schedule="cosine",
                     **config)
    del dummy_env

    # Create variable environment template
    def extra_step_filter(x):
        # If in rectangle
        if config['bottom_bar_coord'] < x[1] < config['top_bar_coord']:
            # with p == 0.05, delay by 20 steps
            if np.random.uniform() < 0.05:
                return 20
        # Otherwise, normal time steps (no delay)
        return 0

    envs = make_vec_env(lambda: make_variable_env('LunarLander-v2', fn=extra_step_filter),
                        n_envs=config['n_envs'])

    # ============================================================== #
    # ========================= TRAINING =========================== #
    # ============================================================== #

    # Train agent
    if train:
        if logging:
            os.environ['WANDB_BASE_URL'] = "http://localhost:8080"
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

        # Sample first batch
        batch, random_key = sampler(agent, key=random_key)

        # Remove anything not needed for jitted training
        excess_data = {}
        removed_data = {}
        for item in ['rewards', 'episode_rewards', 'next_actions', 'next_states', 'dones']:
            excess_data[item] = batch._asdict()[item]
            removed_data[item] = None

        batch = alter_batch(batch, **removed_data)

        # Flatten the batch (optional: downsample if steps specified)
        batch, random_key = flatten_batch(batch, random_key, steps=config['steps'])

        # Train agent
        for epoch in tqdm(range(config['epochs'])):
            actor_loss = 0
            critic_loss = 0

            iteration = 0
            update_iters = 4
            print('\nTraining...')
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
                    loss_info = agent.update(sample,
                                             value_loss_fn={'mse': 0},
                                             actor_loss_fn={'ppo': 0},
                                             clip_ratio=0.2)

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
            batch, random_key = flatten_batch(batch, random_key, config['steps'])

            # Calculate the average reward, log and print it
            average_reward = np.median(excess_data['episode_rewards'])
            print(f'\nEpisode rewards: {average_reward}\n')

            # Checkpoint the model
            if int(average_reward) > best_reward:
                print('Evaluating performance...')
                results = evaluate_envs(agent,
                                        environments=make_vec_env(lambda: make_variable_env('LunarLander-v2',
                                                                                            fn=extra_step_filter),
                                                                  n_envs=1000))
                average_reward = np.median(results)
                print('\n\n', '='*50, f'\nMedian reward: {np.median(results)}, Best reward: {best_reward}\n', '='*50,
                      '\n')
                if int(average_reward) > best_reward:
                    best_reward = int(average_reward)

                    agent.actor.save(os.path.join('./experiments',
                                                  agent.path, f'actor_{best_reward}'))  # if actor else None
                    agent.value.save(os.path.join('./experiments',
                                                  agent.path, f'value_{best_reward}'))  # if value else None

            if logging:
                # Log results
                wandb.log({'actor_loss': actor_loss,
                           'critic_loss': critic_loss,
                           'episode_reward': average_reward})

            if best_reward > 300:
                agent.actor.save(os.path.join('./experiments',
                                              agent.path, f'actor_best'))  # if actor else None
                agent.value.save(os.path.join('./experiments',
                                              agent.path, f'value_best'))
                break

    # ============================================================== #
    # ======================== EVALUATION ========================== #
    # ============================================================== #

    if evaluate:
        # Load the best agent
        filename = agent.path  # './Experiment_2/model_checkpoints'
        agent.actor = agent.actor.load(os.path.join('./experiments', f'{filename}', f'actor_best'))
        agent.value = agent.value.load(os.path.join('./experiments', f'{filename}', f'value_best'))

        # Create the vectorised set of environments
        envs = make_vec_env(lambda: make_variable_env('LunarLander-v2', fn=extra_step_filter),
                            n_envs=5000)

        # Calculate the median reward
        results = evaluate_envs(agent, environments=envs)
        print(f'\nMedian reward: {np.median(results)}')

        # Animate the agent's performance
        env = make_variable_env('LunarLander-v2', fn=extra_step_filter, render_mode='rgb_array')
        run_and_animate(agent, env, runs=20, directory=os.path.join(agent.path, 'gifs'), **config)