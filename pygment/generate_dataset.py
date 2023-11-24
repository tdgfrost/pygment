from gymnasium.envs import make as make_env
import jax
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
import pickle
import os

jax.config.update('jax_platform_name', 'cpu')

# Define config file - could change to FLAGS at some point
if __name__ == "__main__":
    from core.agent import PPOAgent
    from core.common import flatten_batch
    from core.envs import EpisodeGenerator, make_variable_env

    # ============================================================== #
    # ======================== PREPARATION ========================= #
    # ============================================================== #

    # Load previous checkpoints
    reward = 0
    probability = 0.25
    steps = 11
    dirname = f'./experiments/{probability}_probability/PPO/insert_model_dir_here'
    model_checkpoints = os.path.join(dirname, 'model_checkpoints')
    target_directory = f'./offline_datasets/LunarLander/{probability}_probability'

    with open(os.path.join(dirname, 'config.txt'), 'r') as f:
        config = eval(f.read())

    config['max_episode_steps'] = 1000
    config['n_episodes'] = 10000

    # Create agent
    env = make_env('LunarLander-v2', max_episode_steps=config['max_episode_steps'])
    agent = PPOAgent(observations=env.observation_space.sample(),
                     action_dim=env.action_space.n,
                     opt_decay_schedule="cosine",
                     **config)
    del env

    agent.actor = agent.actor.load(os.path.join(model_checkpoints, f'actor_{reward}'))

    # Create variable environment template (optional)
    def extra_step_filter(x):
        # If in rectangle
        if config['bottom_bar_coord'] < x[1] < config['top_bar_coord']:
            # with p == 0.05, delay by 20 steps
            if np.random.uniform() < probability:
                return steps
        # Otherwise, normal time steps (no delay)
        return 0

    envs = make_vec_env(lambda: make_variable_env('LunarLander-v2', fn=extra_step_filter,
                                                  max_episode_steps=config['max_episode_steps']),
                        n_envs=config['n_episodes'])

    # ============================================================== #
    # ====================== GENERATE DATA ========================= #
    # ============================================================== #

    # Create episode generator
    sampler = EpisodeGenerator(envs, gamma=config['gamma'])

    # Generate random key
    random_key = jax.random.PRNGKey(123)

    # Generate batch
    batch, random_key = sampler(agent, key=random_key, verbose=True, max_episode_steps=config['max_episode_steps'])

    # Flatten the batch
    batch = flatten_batch(batch)

    print(f'Average reward of dataset: {np.array(batch.episode_rewards).mean()}')

    # Create directory if it doesn't exist
    os.makedirs(target_directory, exist_ok=True)

    # Save batch
    with open(os.path.join(target_directory, f"dataset_reward_{reward}_{config['n_episodes']}_episodes.pkl"), 'wb') as f:
        pickle.dump(batch, f)
