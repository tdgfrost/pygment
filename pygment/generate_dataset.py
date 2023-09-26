from gymnasium.envs import make as make_env
import jax
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
import pickle
import os

# Define config file - could change to FLAGS at some point
config = {'epochs': int(1e6),
          'seed': 123,
          'n_episodes': 10000,
          'gamma': 0.99,
          'hidden_dims': (64, 64),
          'max_episode_steps': 1000,
          'top_bar_coord': 1.2,  # 0.9,
          'bottom_bar_coord': 0.8,  # 0.5
          }

if __name__ == "__main__":
    from core.agent import PPOAgent
    from core.common import flatten_batch
    from core.envs import EpisodeGenerator, make_variable_env

    # ============================================================== #
    # ======================== PREPARATION ========================= #
    # ============================================================== #

    # Create agent
    env = make_env('LunarLander-v2', max_episode_steps=config['max_episode_steps'])
    agent = PPOAgent(observations=env.observation_space.sample(),
                     action_dim=env.action_space.n,
                     opt_decay_schedule="cosine",
                     **config)
    del env

    # Load previous checkpoints
    reward = 4
    filename = './experiments/PPO/Experiment_1/model_checkpoints'
    agent.actor = agent.actor.load(os.path.join(filename, f'actor_{reward}'))

    # Create variable environment template (optional)
    def extra_step_filter(x):
        # If in rectangle
        if config['bottom_bar_coord'] < x[1] < config['top_bar_coord']:
            # with p == 0.05, delay by 20 steps
            if np.random.uniform() < 0.05:
                return 20
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

    # Create directory if it doesn't exist
    os.makedirs('./offline_datasets/LunarLander/', exist_ok=True)

    # Save batch
    with open(f'./offline_datasets/LunarLander/dataset_reward_{reward}.pkl', 'wb') as f:
        pickle.dump(batch, f)
