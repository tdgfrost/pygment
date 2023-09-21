import jax
import numpy as np
import pickle

if __name__ == "__main__":
    from core.common import Batch

    list_of_batches = []

    for dataset_number, dataset_reward in [[1, -2],
                                           [2, 85],
                                           [3, 214]]:
        with open(f'./offline_datasets/LunarLander/dataset_{dataset_number}/dataset_reward_{dataset_reward}.pkl',
                  'rb') as f:
            list_of_batches.append(pickle.load(f))
            f.close()

    # Downsample batches
    def downsample_batches(batches):
        for idx in range(len(batches)):
            downsampled_batch = batches[idx]._asdict()
            num_episodes = sum(batches[idx].dones)
            shortened_episode_idx = np.where(batches[idx].dones)[0][num_episodes//len(batches)-1]
            for key, val in downsampled_batch.items():
                if val is None:
                    continue
                elif key == 'episode_rewards':
                    downsampled_batch[key] = val[:num_episodes//len(batches)]
                else:
                    downsampled_batch[key] = val[:shortened_episode_idx+1]

            batches[idx] = downsampled_batch

        return batches

    list_of_batches = downsample_batches(list_of_batches)

    # Combine batches
    batch = Batch()._asdict()
    for key in batch.keys():
        for i in range(len(list_of_batches)):
            if batch[key] is None:
                batch[key] = list_of_batches[i][key]
            elif key == 'rewards':
                batch[key].extend(list_of_batches[i][key])
            else:
                batch[key] = np.concatenate((batch[key], list_of_batches[i][key]))

    batch = Batch(**batch)

    with open(f'./offline_datasets/LunarLander/dataset_combined.pkl', 'wb') as f:
        pickle.dump(batch, f)
        f.close()
