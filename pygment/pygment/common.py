import numpy as np
import jax.numpy as jnp
from typing import AnyStr, Dict, Any, List
import flax
from jax import lax
from collections import namedtuple

# Specify types
Params = flax.core.FrozenDict[str, Any]
InfoDict = Dict[str, float]
PRNGKey = Any
Batch = namedtuple('Batch', ['states', 'actions', 'rewards', 'discounted_rewards',
                             'next_states', 'next_actions', 'dones'])


def load_data(path: str,
              scale: str,
              gamma: float = 0.99) -> Dict[AnyStr, np.ndarray]:
    """
    Load data from path
    :param path: Provide a path to the folder containing the data
    :param scale: Dict with keys "normalise", "standardise" or "none"
    :param gamma: Discount factor
    :return: Dict of offline RL data
    """

    # Load the numpy binaries for the offline data
    loaded_data: dict[AnyStr, np.ndarray] = dict()
    for key, filename in [['state', 'all_states.npy'], ['actions', 'all_actions.npy'],
                          ['original_rewards', 'all_rewards.npy'], ['next_state', 'all_next_states.npy'],
                          ['next_action', 'all_next_actions.npy'], ['dones', 'all_dones.npy']]:
        loaded_data[key] = np.load(path + '/' + filename)

    # Pre-process rewards if required (normalise, standardise or none)
    if scale == "standardise":
        loaded_data['rewards'] = (loaded_data['original_rewards'] - loaded_data['original_rewards'].mean()) / \
                                 loaded_data['original_rewards'].std()

    elif scale == "normalise":
        loaded_data['rewards'] = (loaded_data['original_rewards'] - loaded_data['original_rewards'].min()) / (
                    loaded_data['original_rewards'].max() - loaded_data['original_rewards'].min())

    else:
        loaded_data['rewards'] = loaded_data['original_rewards']

    # Calculate discounted rewards
    loaded_data['discounted_rewards'] = calc_discounted_rewards(loaded_data['dones'], loaded_data['rewards'], gamma)

    return loaded_data


def calc_discounted_rewards(dones, rewards, gamma):
    """
    Calculate discounted rewards
    :param dones: Array of step-wise dones
    :param rewards: Array of step-wise rewards
    :param gamma: Discount factor
    :return: Array of discounted rewards
    """
    # Identify the relative length of each trajectory
    idxs = np.where(dones)[0]
    idxs = np.insert(idxs, 0, -1)
    idxs = np.diff(idxs)

    # Identify shape (samples, maximum traj length)
    samples = len(idxs)
    length = np.max(idxs)

    # Create a mask of that shape
    mask = np.zeros((samples, length))
    for sample, end in enumerate(idxs):
        mask[sample, :end] = True
    mask = mask.astype(bool)

    # Create a standard array of that shape, and allocate the rewards using the mask
    discounted_rewards = np.zeros((samples, length))
    discounted_rewards[mask] = rewards

    # Use JAX/lax to calculate the discounted rewards for each row
    discounted_rewards = lax.scan(lambda agg, reward: (gamma * agg + reward, gamma * agg + reward),
                                  np.zeros((samples,)),
                                  jnp.array(discounted_rewards.transpose()), reverse=True)[1].transpose()

    # Finally, convert the array back to a 1D (numpy) array
    discounted_rewards = np.array(discounted_rewards[mask])

    return discounted_rewards


def progress_bar(iteration, total_iterations):
    """
    Print a progress bar to the console
    :param iteration: current iteration
    :param total_iterations: total number of iterations
    :return: None
    """
    bar_length = 30

    percent = iteration / total_iterations
    percent_complete = int(percent * 100)

    progress = int(percent * bar_length)
    progress = '[' + '#' * progress + ' ' * (bar_length - progress) + ']'

    print(f'\r{progress} {percent_complete}% complete', end='', flush=True)
    return

