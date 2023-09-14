import numpy as np
import jax.numpy as jnp
from typing import AnyStr, Dict, Any, List
import flax
from jax import lax
import jax
from collections import namedtuple

# Specify types
Params = flax.core.FrozenDict[str, Any]
InfoDict = Dict[str, Any]
PRNGKey = Any
fields = ['states', 'actions', 'rewards', 'discounted_rewards', 'episode_rewards',
          'next_states', 'next_actions', 'dones', 'action_logprobs', 'advantages']
Batch = namedtuple('Batch', fields, defaults=(None,) * len(fields))


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


def shuffle_split_batch(batch: Batch, steps=1000, batch_size=64):
    # Change batch to list of tuples (agnostic to the labels)
    shuffled_batch = batch._asdict()

    # Create shuffled indexes
    available_steps = 0
    for key, val in shuffled_batch.items():
        if key in ['states', 'next_states', 'actions', 'advantages', 'action_logprobs', 'next_actions', 'dones']:
            available_steps = len(val)
            break

    idxs = np.random.default_rng().choice([i for i in range(available_steps)],
                                          size=(min(steps, available_steps) // batch_size, batch_size),
                                          replace=False)

    # Iterate through and generate each set of shuffled samples
    for sample_idxs in idxs:
        shuffled_batch = batch._asdict()
        # Iterate through each feature and shuffle
        for key, val in shuffled_batch.items():
            if val is None:
                continue
            # Skip the episode rewards (not used for training)
            if key == 'episode_rewards':
                shuffled_batch[key] = val
            # For the rewards, keep as a list of (irregular) lists
            elif key == 'rewards':
                shuffled_batch[key] = [val[i] for i in sample_idxs]
            # Return the remaining features as a NumPy array
            else:
                shuffled_batch[key] = jnp.array(val[sample_idxs]).astype(jnp.float32) if val.dtype == np.float64 \
                    else (jnp.array(val[sample_idxs]).astype(jnp.int32) if val.dtype == np.int64
                          else jnp.array(val[sample_idxs]))
        yield Batch(**shuffled_batch)


def downsample_batch(batch: Batch, random_key, steps):
    flattened_batch = batch._asdict()
    available_steps = 0
    coordinates = {}
    for key, val in flattened_batch.items():
        if key in ['states', 'next_states', 'actions', 'advantages', 'action_logprobs', 'next_actions', 'dones']:
            for i in range(len(val)):
                for j in range(len(val[i])):
                    coordinates[available_steps] = (i, j)
                    available_steps += 1
            break

    flat_idx = np.random.default_rng(int(random_key[0])).choice([i for i in range(available_steps)],
                                                                size=min(steps, available_steps),
                                                                replace=False)

    for key, val in flattened_batch.items():
        if val is None:
            continue
        if key == 'rewards':
            flattened_batch[key] = [val[coordinates[idx][0]][coordinates[idx][1]] for idx in flat_idx]
        elif key == 'episode_rewards':
            flattened_batch[key] = val
        else:
            flattened_batch[key] = np.array([val[coordinates[idx][0]][coordinates[idx][1]] for idx in flat_idx])

    random_key = jax.random.split(random_key, num=1)[0]
    return Batch(**flattened_batch), random_key


def alter_batch(batch, **kwargs):
    """
    Alter the batch by adding or removing features
    :param batch: Batch object
    :param kwargs: Dict of features to add or remove
    :return: Batch object
    """
    # Convert the batch to a dictionary
    batch = batch._asdict()

    # Iterate through the kwargs
    for key, val in kwargs.items():
        batch[key] = val

    # Convert the batch back to a Batch object
    batch = Batch(**batch)

    return batch
