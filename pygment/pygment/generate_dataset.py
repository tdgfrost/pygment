from gymnasium.envs import make as make_env
import jax
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
import pickle

# Define config file - could change to FLAGS at some point
config = {'seed': 123,
          'n_episodes': 10000,
          'gamma': 0.99,
          'hidden_dims': (64, 64),
          }

if __name__ == "__main__":
    from core.agent import PPOAgent
    from core.common import flatten_batch
    from core.envs import EpisodeGenerator, make_variable_env

    # ============================================================== #
    # ======================== PREPARATION ========================= #
    # ============================================================== #

    # Create agent
    env = make_env('LunarLander-v2')
    agent = PPOAgent(observations=env.observation_space.sample(),
                     action_dim=env.action_space.n,
                     opt_decay_schedule="cosine",
                     **config)
    del env

    # Create variable environment template (optional)
    def extra_step_filter(x):
        # If in rectangle
        if config['bottom_bar_coord'] < x[1] < config['top_bar_coord']:
            # with p == 0.05, delay by 20 steps
            if np.random.uniform() < 0.05:
                return 20
        # Otherwise, normal time steps (no delay)
        return 0

    envs = make_vec_env(lambda: make_variable_env('LunarLander-v2', fn=extra_step_filter),
                        n_envs=config['n_episodes'])

    # ============================================================== #
    # ====================== GENERATE DATA ========================= #
    # ============================================================== #

    # Create episode generator
    sampler = EpisodeGenerator(envs, gamma=config['gamma'])

    # Generate random key
    random_key = jax.random.PRNGKey(123)

    # Generate batch
    batch, random_key = sampler(agent, key=random_key)

    # Flatten the batch
    batch, random_key = flatten_batch(batch, random_key)

    # Save batch
    with open('./current_file.pkl', 'wb') as f:
        pickle.dump(batch, f)
