import pygment as pm
import gymnasium as gym
from time import sleep

env = gym.make('LunarLander-v2')
agent = pm.create_agent('policy')
agent.load_env(env)
agent.add_network(nodes=[128, 128])
agent.compile('adam', learning_rate=0.01)
agent.train(target_reward=200, episodes=10000, ep_update=64, gamma=0.99)

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