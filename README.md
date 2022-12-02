# pygment

The purpose of this module is to provide an 'Agent' container for DQN Reinforcement Learning training.

The initial scope of the project will aim to create an agent, which can then load a gymnasium environment; pre-specify a neural network architecture; and train the network on the agent.

A demonstration of the use of this module is shown below:

```
import pygment as pm
import gymnasium as gym


# Create a pygment agent (with its method of learning)
agent = pm.create_agent('DQN')

# Define a gymnasium environment and load it into the agent.
env = gym.make('CartPole-v1', 
                max_episode_steps=500)

agent.load_env(env)

# Define the network architecture and method of learning.
agent.add_network(nodes=[64, 64])
agent.compile(optimizer='adam', learning_rate=0.01)

# And get learning!
agent.train(target_reward=300, 
            save_from=200, 
            save_interval=10,
            episodes=10000,
            gamma=0.99)
```
A print-out of the training will be provided.

The trained network can also be directly invoked by calling:
```
agent.net.forward(obs)
```

Animations can be saved to video, or displayed live, using the animate() or animate_live() methods:
```
pm.animate(agent,
           'CartPole-v1',
           max_episode_steps=500)

pm.animate_live(agent,
                'CartPole-v1',
                max_episode_steps=500)
```
