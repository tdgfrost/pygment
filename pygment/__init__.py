from .agent import DQNAgent, PolicyGradient, ActorCritic
import gymnasium as gym
import os


def create_agent(agent_type='doubleDQN'):
    agent_dict = {'doubleDQN': DQNAgent(),
                  'actorcritic': ActorCritic(),
                  'policy': PolicyGradient()}

    if agent_type not in agent_dict.keys():
        error = 'type must be one of: '
        for key in agent_dict.keys():
            error += key + ', '
        error = error[:-2]
        raise KeyError(error)

    return agent_dict[agent_type]


def animate(agent, env_name, max_episode_steps=500, directory='./videos'):
    if not os.path.isdir(directory):
        os.mkdir(directory)

    env = gym.make(env_name, render_mode='rgb_array', max_episode_steps=max_episode_steps)
    env = gym.wrappers.RecordVideo(env, directory)
    agent.load_env(env)
    agent.net.to('mps')

    done = False
    state = agent.env.reset()[0]
    while not done:
        action = agent.net(state)[0].item()
        next_state, reward, done, _, _ = agent.env.step(action)
        state = next_state

    agent.env.close()


def animate_live(agent, env_name, max_episode_steps=500):
  env = gym.make(env_name, render_mode='human_mode', max_episode_steps=max_episode_steps)
  agent.load_env(env)
  agent.net.to('mps')

  done = False
  state = agent.env.reset()[0]
  self.env.render()
  while not done:
    action = agent.net(state)[0].item()
    next_state, reward, done, _, _ = agent.env.step(action)
    state = next_state
    self.env.render()
