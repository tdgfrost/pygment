from .agent import DQNAgent, PolicyGradient, PPO, PPOContinuous, IQLAgent, Experience
import gymnasium as gym
import os
import torch
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"


def create_agent(agent_type='doubledqn', device='mps', path=None):
    agent_dict = {'doubledqn': DQNAgent(device, path),
                  'ppo': PPO(device, path),
                  'policy': PolicyGradient(device, path),
                  'ppocontinuous': PPOContinuous(device, path),
                  'iql': IQLAgent(device, path)}

    if agent_type.lower() not in agent_dict.keys():
        error = 'type must be one of: '
        for key in agent_dict.keys():
            error += key + ', '
        error = error[:-2]
        raise KeyError(error)

    return agent_dict[agent_type.lower()]


def animate(agent_instance, env_name, max_episode_steps=1000, directory=None, prefix='', target_reward=None):
    if directory is None:
        directory = agent_instance.path + '/videos'

    if not os.path.isdir(directory):
        os.makedirs(directory)

    # agent_instance.net.to('mps')
    if target_reward is not None:
        counter = 0
        while True:
            counter += 1
            env = gym.make(env_name, render_mode='rgb_array', max_episode_steps=max_episode_steps)
            next_video = 0
            for file in os.listdir(directory):
                if file.endswith('.mp4'):
                    next_video += 1
            env = gym.wrappers.RecordVideo(env, directory, name_prefix=f'{prefix}_video{next_video}')
            agent_instance.load_env(env)
            if type(env.action_space) == gym.spaces.discrete.Discrete:
                continuous = False
            else:
                continuous = True

            total_rewards = 0
            done = False
            truncated = False
            state = agent_instance.env.reset()[0]
            while not done and not truncated:
                if continuous:
                    action_means, action_stds, _ = agent_instance.net(state)
                    action, _, _ = agent_instance.get_action_and_logprobs(action_means, action_stds)
                    action = torch.tanh(action).numpy()
                else:
                    action = agent_instance.choose_action(state)
                next_state, reward, done, truncated, _ = agent_instance.env.step(action)
                state = next_state
                total_rewards += reward

            agent_instance.env.close()

            if 0.98 * target_reward <= total_rewards <= 1.02 * target_reward:
                for file in os.listdir(directory):
                    if file.endswith('.meta.json'):
                        os.remove(os.path.join(directory, file))
                    if file.endswith('-episode-0.mp4'):
                        os.rename(os.path.join(directory, file), os.path.join(directory, f'{prefix}_video{next_video}.mp4'))
                break
            else:
                del agent_instance.env
                for file in os.listdir(directory):
                    if file.endswith('.meta.json'):
                        os.remove(os.path.join(directory, file))
                    if file.endswith('-episode-0.mp4'):
                        os.remove(os.path.join(directory, file))
                if counter == 30:
                    print('30 attempts made - unable to capture target reward')
                    break


def animate_live(agent_instance, env_name, max_episode_steps=500):
    env = gym.make(env_name, render_mode='human', max_episode_steps=max_episode_steps)
    agent_instance.load_env(env)
    agent_instance.net.to('mps')

    done = False
    state = agent_instance.env.reset()[0]
    agent_instance.env.render()
    while not done:
        action = agent_instance.net(state)[0].item()
        next_state, _, done, _, _ = agent_instance.env.step(action)
        state = next_state
        agent_instance.env.render()
