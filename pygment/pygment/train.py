from tqdm import tqdm
import numpy as np
import gymnasium
from tensorboardX import SummaryWriter
import os
import jax
import jax.numpy as jnp
import ray

# Set jax to CPU
jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)
# jax.config.update('jax_disable_jit', True)

# Define config file - could change to FLAGS at some point
config = {'seed': 123,
          'epochs': int(1e6),
          'batch_size': int(1e5),
          'expectile': 0.8,
          'gamma': 0.9999,
          'actor_lr': 5e-3,
          'value_lr': 5e-3,
          'critic_lr': 5e-3,
          'hidden_dims': (256, 256),
          'clipping': 1,
          }


if __name__ == "__main__":
    from agent import IQLAgent
    from common import load_data, Batch

    # Set whether to train and/or evaluate
    train = False
    evaluate = True

    # Create environment
    env = gymnasium.envs.make('LunarLander-v2', max_episode_steps=1000)

    # Load static dataset (dictionary) and convert to a 1D list of Experiences
    data = load_data(path='../samples/GenerateStaticDataset/LunarLander/140 reward',
                     scale='standardise',
                     gamma=config['gamma'])

    data = Batch(states=data['state'],
                 actions=data['actions'][:, np.newaxis],
                 rewards=data['rewards'],
                 discounted_rewards=data['discounted_rewards'],
                 next_states=data['next_state'],
                 next_actions=data['next_action'][:, np.newaxis],
                 dones=data['dones'])

    # Create agent
    agent = IQLAgent(observations=env.observation_space.sample(),
                     actions=env.action_space.sample()[np.newaxis],
                     action_dim=env.action_space.n,
                     dropout_rate=None,
                     opt_decay_schedule="cosine",
                     **config)

    # Prepare logging tensorboard
    summary_writer = SummaryWriter('../experiments/tensorboard/current',
                                   write_to_disk=True)
    os.makedirs('../experiments/tensorboard/current/', exist_ok=True)

    # Train agent
    if train:
        for value, critic, actor in [[True, False, False], [False, True, False], [False, False, True]]:

            loss_key = f"{'value' if value else ('critic' if critic else 'actor')}_loss"
            best_loss = jnp.inf
            count = 0
            for epoch in tqdm(range(config['epochs'])):
                batch = agent.sample(data,
                                     config['batch_size'])

                loss_info = agent.update_async(batch, actor, critic, value)

                # Record best loss
                if loss_info[loss_key] < best_loss:
                    best_loss = loss_info[loss_key]
                    count = 0
                    agent.actor.save(os.path.join('../experiments', agent.path, 'actor')) if actor else None
                    agent.critic.save(os.path.join('../experiments', agent.path, 'critic')) if critic else None
                    agent.value.save(os.path.join('../experiments', agent.path, 'value')) if value else None
                else:
                    count += 1
                    if count > 1000:
                        agent.actor = agent.actor.load(
                            os.path.join('../experiments', agent.path, 'actor')) if actor else agent.actor
                        agent.critic = agent.critic.load(
                            os.path.join('../experiments', agent.path, 'critic')) if critic else agent.critic
                        agent.value = agent.value.load(
                            os.path.join('../experiments', agent.path, 'value')) if value else agent.value
                        break

                # Log intermittently
                if epoch % 5 == 0:
                    for key, val in loss_info.items():
                        if key == 'layer_outputs':
                            continue
                        if val.ndim == 0:
                            summary_writer.add_scalar(f'training/{key}', val, epoch)
                    summary_writer.flush()

    """
    Time to evaluate!
    """
    if evaluate:
        agent.actor = agent.actor.load(os.path.join('../experiments/experiment_3', 'actor'))
        agent.critic = agent.critic.load(os.path.join('../experiments/experiment_3', 'critic'))
        agent.value = agent.value.load(os.path.join('../experiments/experiment_3', 'value'))

        ray.init()

        @ray.remote
        def evaluate_policy(policy, environment):
            done, prem_done = False, False
            total_reward = 0
            key = jax.random.PRNGKey(123)
            device = jax.devices('cpu')[0]

            state, _ = environment.reset()
            state = jax.device_put(jnp.array(state), device)

            while not done and not prem_done:
                _, logits = policy(state)
                action = jax.random.categorical(key, logits).item()

                state, reward, done, prem_done, _ = environment.step(action)

                total_reward += reward
                key = jax.random.split(key, num=1)[0]
                state = jax.device_put(state, device)

            return total_reward


        # Define the number of environments to evaluate in parallel
        parallel_envs = 512

        import multiprocessing
        from functools import partial

        def evaluate_policy(environment, policy):
            device = jax.devices('cpu')[0]
            key = jax.random.PRNGKey(123)

            total_reward = 0
            state, _ = environment.reset()
            state = jax.device_put(state, device)
            done, prem_done = False, False

            while not done and not prem_done:
                _, logits = policy(state)
                action = jax.random.categorical(key, logits).item()

                state, reward, done, prem_done, _ = environment.step(action)

                total_reward += reward
                key = jax.random.split(key, num=1)[0]
                state = jax.device_put(state, device)

            return total_reward

        envs = [gymnasium.make('LunarLander-v2', max_episode_steps=500) for _ in range(parallel_envs)]

        with multiprocessing.Pool(processes=parallel_envs) as pool:
            evaluate_partial = partial(evaluate_policy, policy=agent.actor)

            rewards = pool.map(evaluate_partial, envs)

        for env in envs:
            env.close()



        total_rewards = ray.get([evaluate_policy.remote(agent.actor,
                                                        gymnasium.envs.make('LunarLander-v2',
                                                                            max_episode_steps=500))
                                 for _ in range(parallel_envs)])

        print(f'Total rewards: {np.array(total_rewards).mean()}')
