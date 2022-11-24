import pygment as pm
import gymnasium as gym
from time import sleep
import torch

env = gym.make('LunarLander-v2')
agent = pm.create_agent('policy')
agent.load_env(env)
agent.add_network(nodes=[32, 32, 32])
agent.compile('adam', learning_rate=0.001)
agent.train(target_reward=200, episodes=100000, ep_update=32, gamma=0.99, max_steps=1000)

env = gym.make('LunarLander-v2', render_mode='human')
agent.load_env(env)
done = False
state = agent.env.reset()[0]
while not done:
    agent.env.render()
    action = agent.action_selector(state)
    next_state, reward, done, _, _ = agent.env.step(action)
    state = next_state
    #sleep(0.2)

if False:
    env = gym.make('CartPole-v1')
    agent.load_env(env)
    agent.train(500, target_update=500, epsilon=0.02)

# Code for animation
if False:
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(env, '/Users/thomasfrost/Documents/Github/pygment/pygment/pygment/samples/LunarLander-v2-policy-gradient/videos')
    agent.load_env(env)

    done = False
    state = agent.env.reset()[0]
    while not done:
        #action = agent.action_selector(state)
        action = agent.net(state)[0]
        next_state, reward, done, _, _ = agent.env.step(action)
        state = next_state

    agent.env.close()
