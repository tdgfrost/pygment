from .agent import DQNAgent, PolicyGradient, PPO, PPOContinuous, IQLAgent, Experience
import gymnasium as gym
import os
import torch


def create_agent(agent_type='doubledqn', device='mps'):
    agent_dict = {'doubledqn': DQNAgent(device),
                  'ppo': PPO(device),
                  'policy': PolicyGradient(device),
                  'ppocontinuous': PPOContinuous(device),
                  'iql': IQLAgent(device)}

    if agent_type.lower() not in agent_dict.keys():
        error = 'type must be one of: '
        for key in agent_dict.keys():
            error += key + ', '
        error = error[:-2]
        raise KeyError(error)

    return agent_dict[agent_type.lower()]


def animate(agent_instance, env_name, max_episode_steps=500, directory=None, prefix=''):
    if directory is None:
        directory = agent_instance.path + '/videos'

    # agent_instance.net.to('mps')

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

    done = False
    state = agent_instance.env.reset()[0]
    while not done:
        if continuous:
            action_means, action_stds, _ = agent_instance.net(state)
            action, _, _ = agent_instance.get_action_and_logprobs(action_means, action_stds)
            action = torch.tanh(action).numpy()
        else:
            action = agent_instance.choose_action(state)
        next_state, _, done, _, _ = agent_instance.env.step(action)
        state = next_state

    agent_instance.env.close()
    for file in os.listdir(directory):
        if file.endswith('.meta.json'):
            os.remove(os.path.join(directory, file))
        if file.endswith('-episode-0.mp4'):
            os.rename(os.path.join(directory, file), os.path.join(directory, f'{prefix}_video{next_video}.mp4'))


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
