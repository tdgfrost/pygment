import pygment as pm
import gymnasium as gym
from time import sleep

env = gym.make('LunarLander-v2')
agent = pm.create_agent('actorcritic')
agent.load_env(env)
agent.add_layer(64, 'relu')
agent.add_layer(64, 'relu')
agent.add_layer(64, 'relu')
agent.compile('adam', learning_rate=0.1)
agent.train(200, decay_rate=0.995, gamma=0.99, min_epsilon=0.02, batch_size=512,
            tau=0.001)

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