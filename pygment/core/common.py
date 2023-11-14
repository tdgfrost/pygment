import numpy as np
import jax.numpy as jnp
from typing import Dict, Any
import flax
import jax
from collections import namedtuple
import pickle

# Specify types
Params = flax.core.FrozenDict[str, Any]
InfoDict = Dict[str, Any]
PRNGKey = Any
fields = ['states', 'actions', 'rewards', 'discounted_rewards', 'episode_rewards',
          'next_states', 'next_actions', 'dones', 'action_logprobs', 'advantages', 'intervals',
          'len_actions', 'next_len_actions']
Batch = namedtuple('Batch', fields, defaults=(None,) * len(fields))


def load_data(path: str,
              scale='none',
              gamma: float = 0.99) -> Batch:
    """
    Load data from path
    :param path: Provide a path to the folder containing the data
    :param scale: Dict with keys "normalise", "standardise" or "none"
    :param gamma: Discount factor
    :return: Dict of offline RL data
    """

    # Load the saved Batch
    with open(path, 'rb') as f:
        batch = pickle.load(f)
    """
    loaded_data: dict[AnyStr, np.ndarray] = dict()
    for key, filename in [['state', 'all_states.npy'], ['actions', 'all_actions.npy'],
                          ['original_rewards', 'all_rewards.npy'], ['next_state', 'all_next_states.npy'],
                          ['next_action', 'all_next_actions.npy'], ['dones', 'all_dones.npy']]:
        loaded_data[key] = np.load(path + '/' + filename)
    """

    # Pre-process rewards if required (normalise, standardise or none)
    if scale == "standardise":
        rewards = np.concatenate(batch.rewards)
        rewards_mean = rewards.mean()
        rewards_std = np.maximum(rewards.std(), 1e-8)
        rewards = [[(r - rewards_mean) / rewards_std for r in seq] for seq in batch.rewards]

        discounted_rewards = calc_discounted_rewards(batch.dones, rewards, gamma)

        batch = alter_batch(batch, rewards=rewards, discounted_rewards=discounted_rewards)

    elif scale == "normalise":
        rewards = np.concatenate(batch.rewards)
        rewards_max = rewards.max()
        rewards_min = rewards.min()
        rewards = [[(r - rewards_min) / (rewards_max - rewards_min) for r in seq] for seq in batch.rewards]

        discounted_rewards = calc_discounted_rewards(batch.dones, rewards, gamma)

        batch = alter_batch(batch, rewards=rewards, discounted_rewards=discounted_rewards)

    # Calculate discounted rewards
    if batch.discounted_rewards is None:
        discounted_rewards = calc_discounted_rewards(batch.dones, batch.rewards, gamma)
        batch = alter_batch(batch, discounted_rewards=discounted_rewards)

    return batch


def move_to_gpu(batch: Batch, gpu_keys: list, gpu_key='gpu') -> Batch:
    gpu_batch = batch._asdict()

    for key, value in gpu_batch.items():
        if value is None:
            continue
        if key in gpu_keys:
            if type(value) == jnp.ndarray:
                gpu_batch[key] = jax.device_put(value, jax.devices(gpu_key)[0])
            else:
                gpu_batch[key] = jnp.array(value)

    return Batch(**gpu_batch)


def calc_discounted_rewards(dones, rewards, gamma):
    """
    Calculate discounted rewards
    :param dones: Array of step-wise dones
    :param rewards: Array of step-wise rewards
    :param gamma: Discount factor
    :return: Array of discounted rewards
    """
    # Create the relevant indexes for various upcoming tasks:
    #   - action_idxs: the index of the first step in each episode (subdivided by actions)
    action_idxs = np.where(dones)[0] + 1
    action_idxs = np.insert(action_idxs, 0, 0)

    #   - episode_lengths: the length of each episode (subdivided into environment steps)
    episode_lengths = np.array([sum([len(traj) for traj in rewards[start:end]])
                                for start, end in zip(action_idxs[:-1], action_idxs[1:])])

    #   - traj_idxs: the index of each action taken, relative to each step of the environment
    traj_lengths = np.array([len(traj) for traj in rewards])
    traj_lengths = np.insert(traj_lengths, 0, 0)
    traj_idxs = np.cumsum(traj_lengths)[:-1]

    # Create a zero-padded array of the rewards
    #   - Start by finding the number of samples and maximum length of an episode (subdivided into environment steps)
    samples = len(episode_lengths)
    length = np.max(episode_lengths)

    #   - Create a mask of that shape
    mask = np.zeros((samples, length))
    for sample, end in enumerate(episode_lengths):
        mask[sample, :end] = True
    mask = mask.astype(bool)

    #   - Create a zero-padded array of the rewards using the above mask
    discounted_rewards = np.zeros((samples, length))
    flattened_rewards = np.concatenate(rewards)
    discounted_rewards[mask] = flattened_rewards

    # Use JAX/lax to calculate the discounted rewards for each row
    def scan_fn(accumulator, current): return gamma * accumulator + current
    accumulator = np.frompyfunc(scan_fn, 2, 1)
    discounted_rewards = np.flip(accumulator.accumulate(np.flip(discounted_rewards, -1), axis=-1), -1).astype(np.float32)

    # Finally, convert the array back to a 1D (numpy) array
    discounted_rewards = discounted_rewards[mask]

    # And index by the actions taken, not the environment steps
    discounted_rewards = discounted_rewards[traj_idxs]

    return discounted_rewards


def calc_traj_discounted_rewards(rewards, gamma):
    max_len = max([len(traj) for traj in rewards])
    samples = len(rewards)

    # Define the 'empty' array of boolean masks
    mask = np.zeros(shape=(len(rewards), max_len),
                    dtype=np.bool_)

    # Identify the 'end point' of each trajectory
    idxs = np.column_stack((np.arange(len(rewards)), [len(traj) - 1 for traj in rewards]))
    mask[idxs[:, 0], idxs[:, 1]] = True

    # Create an accumulator function, which will backtrace the "True" mask label back to the first index for those rows
    def scan_fn(accumulator, current): return max(accumulator, current)
    accumulator = np.frompyfunc(scan_fn, 2, 1)

    # Apply only to the rows where the mask isn't in the first position to begin with
    mask[np.where(np.where(mask)[1] > 0)[0]] = np.flip(accumulator.accumulate(
        np.flip(mask[np.where(np.where(mask)[1] > 0)[0]], -1), axis=-1), -1)

    # Create a zero-padded array of the rewards using the above mask
    discounted_rewards = np.zeros((samples, max_len))
    discounted_rewards[mask] = np.concatenate(rewards)

    # Create the gammas and apply to the discounted rewards
    gammas = np.ones(shape=max_len) * gamma
    gammas = np.power(gammas, np.arange(max_len))

    # Sum up each trajectory into a single value
    discounted_rewards = np.sum(discounted_rewards * gammas, axis=-1)

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


def shuffle_split_batch(batch: Batch, batch_size=64):
    # Change batch to list of tuples (agnostic to the labels)
    shuffled_batch = batch._asdict()

    # Create shuffled indexes
    available_steps = 0
    for key, val in shuffled_batch.items():
        if key in ['states', 'next_states', 'actions', 'advantages', 'action_logprobs', 'next_actions', 'dones']:
            available_steps = len(val)
            break

    idxs = np.random.default_rng().choice([i for i in range(available_steps)],
                                          size=(available_steps // batch_size, batch_size),
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


def flatten_batch(batch: Batch):
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

    flat_idx = np.array([i for i in range(available_steps)])

    for key, val in flattened_batch.items():
        if val is None:
            continue
        if key == 'rewards':
            flattened_batch[key] = [val[coordinates[idx][0]][coordinates[idx][1]] for idx in flat_idx]
        elif key == 'episode_rewards':
            flattened_batch[key] = val
        else:
            flattened_batch[key] = np.array([val[coordinates[idx][0]][coordinates[idx][1]] for idx in flat_idx])

    return Batch(**flattened_batch)


def downsample_batch(batch: Batch, random_key, steps=None):
    downsampled_batch = batch._asdict()
    available_steps = len(downsampled_batch['states'])

    if steps is None:
        return Batch(**downsampled_batch), random_key

    else:
        flat_idx = np.random.default_rng(int(random_key[0])).choice([i for i in range(available_steps)],
                                                                    size=min(steps, available_steps),
                                                                    replace=False)

    for key, val in downsampled_batch.items():
        if val is None:
            continue
        elif isinstance(val, list):
            downsampled_batch[key] = [val[idx] for idx in flat_idx]
        elif key == 'episode_rewards':
            downsampled_batch[key] = val
        else:
            downsampled_batch[key] = val[flat_idx]

    random_key = jax.random.split(random_key, num=1)[0]
    return Batch(**downsampled_batch), random_key


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


def filter_to_action(array, actions):
    return jax.lax.gather(array,
                          jnp.concatenate((jnp.arange(len(actions)).reshape(-1, 1),
                                           actions.reshape(-1, 1)), axis=1),
                          jax.lax.GatherDimensionNumbers(
                              offset_dims=tuple([]),
                              collapsed_slice_dims=tuple([0, 1]),
                              start_index_map=tuple([0, 1])), tuple([1, 1]))


def filter_dataset(batch: Batch, boolean_identity, target_keys: list):
    filtered_batch = batch._asdict()

    for key, val in filtered_batch.items():
        if key in target_keys:
            if val is None:
                continue
            if type(val) == list:
                filtered_batch[key] = [item for idx, item in zip(boolean_identity, val) if idx]
            else:
                filtered_batch[key] = val[boolean_identity]

    return Batch(**filtered_batch)


def split_output(output, dim=2):
    result = []
    original_shape = output.shape[:-1]
    for i in range(dim):
        result.append(output.reshape(-1, dim)[:, i].reshape(original_shape))
    return tuple(result)
