import numpy as np
from gymnasium.envs import make as make_env
import os
import jax.numpy as jnp
from stable_baselines3.common.env_util import make_vec_env

# Define config file - could change to FLAGS at some point
config = {'seed': 123,
          'gamma': 0.99,
          'hidden_dims': (64, 64),
          'clipping': 1,
          'top_bar_coord': 1.2,  # 0.9,
          'bottom_bar_coord': 0.8,  # 0.5
          }

if __name__ == "__main__":
    from core.agent import IQLAgent
    from core.common import load_data
    from core.evaluate import evaluate_envs, run_and_animate
    from core.envs import make_variable_env

    # ============================================================== #
    # ========================= TRAINING =========================== #
    # ============================================================== #

    # Load static dataset
    print('Loading and processing dataset...')
    baseline_reward = -2
    data = load_data(path=f'./offline_datasets/LunarLander/dataset_1/dataset_reward_{baseline_reward}.pkl',
                     scale='standardise',
                     gamma=config['gamma'])

    intervals = jnp.array([len(traj) for traj in data.rewards])
    interval_range = intervals.max() - intervals.min() + 1  # Range is inclusive so add 1 to this number

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
    # ======================== EVALUATION ========================== #
    # ============================================================== #

    dummy_env = make_env('LunarLander-v2')

    agent = IQLAgent(observations=dummy_env.observation_space.sample(),
                     action_dim=dummy_env.action_space.n,
                     interval_dim=interval_range.item(),
                     interval_min=intervals.min().item(),
                     opt_decay_schedule='none',
                     **config)

    del dummy_env

    agent.standardise_inputs(data.states)

    desired_path = input('Enter the name of the IQL model you wish to evaluate: ')
    gif_bool = input('Do you want to generate gifs? (y/n): ') == 'y'

    model_dir = os.path.join('./experiments/IQL', desired_path)
    agent.actor = agent.actor.load(os.path.join(model_dir, 'model_checkpoints/actor'))

    max_episode_steps = 1000
    envs_to_evaluate = 1000

    print('\n\n', '=' * 50, '\n', ' ' * 3, '\U0001F514' * 3, ' ' * 1, f'Evaluating network', ' ' * 2,
          '\U0001F514' * 3, '\n', '=' * 50)

    """
    TEMPORARY CODE
    """
    env = make_variable_env('LunarLander-v2', fn=extra_step_filter, render_mode='rgb_array')
    import os
    import jax
    import numpy as np
    from matplotlib.patches import Rectangle
    from core.evaluate import animate_blocked_environment

    def run_and_animate(policy,
                        environment,
                        random_key=jax.random.PRNGKey(123),
                        directory='./experiments/IQL/MagiMixerExperiment_1/temp_gifs',
                        *args, **kwargs):

        # Set the parameters for the Rectangle to be plotted
        top = 1.2
        bottom = 0.8
        img_grad = 300 / 1.5
        width = (top - bottom) * img_grad

        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Define the random key
        random_key = jax.random.split(random_key, num=1)[0]

        count = 0
        while True:
            count += 1
            print(f'Count: {count}')
            # Define the empty list of frames
            img_arrays = []
            rewards = []
            seq_rewards = []

            # Iterate over the environment, saving all the frames as rgb_arrays
            state, _ = environment.reset()
            done, prem_done = False, False
            img_arrays.append(environment.render())
            while not done and not prem_done:
                action = policy.sample_action(state, random_key)[0]
                state, reward, done, prem_done, info = environment.step(action)
                img_arrays.extend(info['render_arrays'])
                rewards.append(reward)
                seq_rewards.append(info['sequential_reward'])

            if np.array([len(traj) for traj in seq_rewards]).max() > 1:
                break

        # Define the Rectangle and animate the environment with it
        rect = Rectangle((0, (300 - img_grad * top)),
                         600,
                         width,
                         linewidth=1,
                         edgecolor='r',
                         facecolor='none')

        animate_blocked_environment(img_arrays,
                                    os.path.join(directory, 'gif.gif'),
                                    patch=rect,
                                    fps=environment.metadata['render_fps'])

        # Close the environment
        environment.close()
        return np.array(rewards), seq_rewards

    rewards, seq_rewards = run_and_animate(agent, env)

    results = evaluate_envs(agent, make_vec_env(lambda: make_variable_env('LunarLander-v2',
                                                                          fn=extra_step_filter),
                                                n_envs=envs_to_evaluate))
    print(f'\nMedian reward: {np.median(results)}')
    print(f'Mean reward: {np.mean(results)}')

    # Animate the agent's performance
    if gif_bool:
        print('\n\n', '=' * 50, '\n', ' ' * 3, '\U0001F4FA' * 3, ' ' * 1, f'Generating gifs', ' ' * 2,
              '\U0001F4FA' * 3, '\n', '=' * 50)
        env = make_variable_env('LunarLander-v2', fn=extra_step_filter, render_mode='rgb_array')
        run_and_animate(agent, env, runs=20, directory=os.path.join(model_dir, 'gifs'), **config)
