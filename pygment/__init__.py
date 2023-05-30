from .agent import DQNAgent, PolicyGradient, PPO, PPOContinuous
import gymnasium as gym
import os
import torch


def create_agent(agent_type='doubledqn', device='mps'):
    agent_dict = {'doubledqn': DQNAgent(device),
                  'ppo': PPO(device),
                  'policy': PolicyGradient(device),
                  'ppocontinuous': PPOContinuous(device)}

    if agent_type.lower() not in agent_dict.keys():
        error = 'type must be one of: '
        for key in agent_dict.keys():
            error += key + ', '
        error = error[:-2]
        raise KeyError(error)

    return agent_dict[agent_type.lower()]


def animate(agent, env_name, max_episode_steps=500, directory=None):
    if directory is None:
      directory = agent._path + '/videos'

    env = gym.make(env_name, render_mode='rgb_array', max_episode_steps=max_episode_steps)
    env = gym.wrappers.RecordVideo(env, directory)
    agent.load_env(env)
    #agent.net.to('mps')
    if type(env.action_space) == gym.spaces.discrete.Discrete:
        continuous = False
    else:
        continuous = True

    done = False
    state = agent.env.reset()[0]
    while not done:
        if continuous:
          action_means, action_stds, _ = agent.net(state)
          action, _, _ = agent.get_action_and_logprobs(action_means, action_stds)
          action = torch.tanh(action).numpy()
        else:
          action = agent.net(state)[0].item()
        next_state, _, _, _, _ = agent.env.step(action)
        state = next_state

    agent.env.close()


def animate_live(agent, env_name, max_episode_steps=500):
  env = gym.make(env_name, render_mode='human', max_episode_steps=max_episode_steps)
  agent.load_env(env)
  agent.net.to('mps')

  done = False
  state = agent.env.reset()[0]
  agent.env.render()
  while not done:
    action = agent.net(state)[0].item()
    next_state, reward, done, _, _ = agent.env.step(action)
    state = next_state
    agent.env.render()
