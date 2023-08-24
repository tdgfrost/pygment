import numpy as np
import jax.numpy as jnp
from typing import AnyStr, Dict


class Experience:
    """
    The Experience class stores values from a training episode (values from the environment
    and values from the network).
    """

    def __init__(self,
                 state=None,
                 action=None,
                 reward=None,
                 discounted_reward=None,
                 next_state=None,
                 next_action=None,
                 done=None,
                 original_reward=None,
                 original_discounted_reward=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.discounted_reward = discounted_reward
        self.next_state = next_state
        self.next_action = next_action
        self.done = done
        self.original_reward = original_reward if original_reward is not None else reward
        self.original_discounted_reward = original_discounted_reward if original_discounted_reward is not None else discounted_reward


def load_data(path: str,
              scale: str,
              gamma: float = 0.99) -> Dict[AnyStr]:
    """
    Load data from path
    :param path: Provide a path to the folder containing the data
    :param scale: Dict with keys "normalise", "standardise" or "none"
    :param gamma: Discount factor
    :return: Dict of offline RL data
    """

    # Load the numpy binaries for the offline data
    loaded_data: dict[AnyStr, jnp.ndarray] = dict()
    for key, filename in [['state', 'all_states.npy'], ['actions', 'all_actions.npy'],
                          ['original_rewards', 'all_rewards.npy'], ['next_state', 'all_next_states.npy'],
                          ['next_action', 'all_next_actions.npy'], ['dones', 'all_dones.npy']]:
        loaded_data[key] = jnp.load(path + '/' + filename)

    # Pre-process rewards if required (normalise, standardise or none)
    if scale == "standardise":
        loaded_data['rewards'] = (loaded_data['original_rewards'] - loaded_data['original_rewards'].mean()) / loaded_data['original_rewards'].std()

    elif scale == "normalise":
        loaded_data['rewards'] = (loaded_data['original_rewards'] - loaded_data['original_rewards'].min()) / (loaded_data['original_rewards'].max() - loaded_data['original_rewards'].min())

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
    # Create a list of tuples containing the start and end indices of each trajectory
    idxs = (np.where(dones)[0] + 1).tolist()
    idxs.insert(0, 0)
    idxs = np.array([(start_idx, end_idx) for start_idx, end_idx in zip(idxs[:-1], idxs[1:])])

    # Iterate over each tuple of trajectories and calculate the discounted rewards
    discounted_rewards = []

    for start_idx, end_idx in idxs:
        discounted_reward = 0
        for idx in range(end_idx-1, start_idx-1, -1):
            discounted_reward = rewards[idx] + gamma * discounted_reward
            discounted_rewards.append(discounted_reward)

    return jnp.array(discounted_rewards)
