import os
import jax
import numpy as np
import gymnasium
from stable_baselines3.common.env_util import make_vec_env
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from core.common import progress_bar
from typing import List


def evaluate_envs(policy, environments):
    """
    Evaluate the agent across vectorised episodes.

    :param policy: policy to evaluate.
    :param environments: envs to evaluate.
    :return: array of total rewards for each episode.
    """
    nodes = environments.num_envs
    # Initial parameters
    key = jax.random.PRNGKey(123)
    states = environments.reset()
    dones = np.array([False for _ in range(nodes)])
    idxs = np.array([i for i in range(nodes)])
    all_rewards = np.array([0. for _ in range(nodes)])

    while not dones.all():
        progress_bar(dones.sum(), nodes)
        # Step through environments
        actions = policy.sample_action(states, key)[0]
        states, rewards, new_dones, prem_dones = environments.step(actions)

        # Update finished environments
        prem_dones = np.array([d['TimeLimit.truncated'] for d in prem_dones])
        dones[idxs] = np.any((new_dones, prem_dones), axis=0)[idxs]

        # Update rewards
        all_rewards[idxs] += np.array(rewards)[idxs]

        # Update remaining parameters
        idxs = np.where(~dones)[0]
        states = np.array(states)
        key = jax.random.split(key, num=1)[0]

    return all_rewards


def animate_blocked_environment(img_arrays: List[np.ndarray], save_path: str, patch, fps: int = 50):

    # (0,0) is where the flags are centered.
    # On the plot, this is based as (300, 300), with (0, 300) at the top
    # So to calculate the correct y_plot, the formula is y_plot = 300 - (300 / 1.5) * y_state

    # Define the function that will iterate over each frame to update
    def update(i):
        im.set_array(img_arrays[i])
        return im,

    # Define the plot - remove axes and keep it tight (no white space padding)
    fig, ax = plt.subplots()
    plt.axis('off')
    plt.tight_layout()

    # Plot the first frame and define as animated
    im = ax.imshow(img_arrays[0], animated=True)

    # Add the Rectangle patch to the plot
    ax.add_patch(patch)

    # Iterate over each frame and update the plot
    animation_fig = FuncAnimation(fig, update, frames=len(img_arrays), interval=1000 / fps,
                                  cache_frame_data=False)

    # Save as a .gif in the specified path
    animation_fig.save(save_path)

    # Close the plot
    plt.close()


def run_and_animate(policy,
                    environment,
                    runs=20,
                    random_key=jax.random.PRNGKey(123),
                    directory='./experiments/gifs',
                    *args, **kwargs):

    # Set the parameters for the Rectangle to be plotted
    top = kwargs['top_bar_coord']
    bottom = kwargs['bottom_bar_coord']
    img_grad = 300 / 1.5
    width = (top - bottom) * img_grad

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Then render as many times as required
    for count in range(runs):
        progress_bar(count, runs)

        # Define the empty list of frames
        img_arrays = []

        # Define the random key
        random_key = jax.random.split(random_key, num=1)[0]

        # Iterate over the environment, saving all the frames as rgb_arrays
        state, _ = environment.reset()
        done, prem_done = False, False
        img_arrays.append(environment.render())
        while not done and not prem_done:
            action = policy.sample_action(state, random_key)[0]
            state, _, done, prem_done, info = environment.step(action)
            img_arrays.extend(info['render_arrays'])

        # Define the Rectangle and animate the environment with it
        rect = Rectangle((0, (300 - img_grad * top)),
                         600,
                         width,
                         linewidth=1,
                         edgecolor='r',
                         facecolor='none')

        animate_blocked_environment(img_arrays,
                                    os.path.join(directory, f'gif{count}.gif'),
                                    patch=rect,
                                    fps=environment.metadata['render_fps'])

    # Close the environment
    environment.close()
