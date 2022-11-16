# pygment

The purpose of this module is to provide an 'Agent' container for DQN Reinforcement Learning training.

The initial scope of the project will aim to create an agent, which can then load a gymnasium environment; pre-specify a neural network architecture; and train the network on the agent.

A demonstration of the use of this module is shown below:

```
import pygment as pm
import gymnasium as gym

agent = pm.create_agent()
env = gym.make('CartPole-v1')
agent.load_env(env)
agent.add_layer(64, 'relu')
agent.add_layer(256, 'relu')
agent.compile(optimizer='adam', learning_rate=0.001)
agent.train()
```
A print-out of the training will be provided.

The trained network can then be invoked by calling:
```
agent.net.main_net(obs)
```
