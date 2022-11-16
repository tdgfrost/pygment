import pygment as pm
import gymnasium as gym

env = gym.make('CartPole-v1')
agent = pm.create_agent()
agent.load_env(env)
agent.add_layer(64, 'relu')
agent.compile('adam', learning_rate=0.01)
agent.train(300, target_update=500)

done = False
state = agent.env.reset()[0]
while not done:
    agent.env.render()
    action = agent.action_selector(state)
    next_state, reward, done, _, _ = agent.env.step(action)
    state = next_state