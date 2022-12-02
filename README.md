# pygment

The purpose of this module is to provide an 'Agent' container for DQN Reinforcement Learning training.

The initial scope of the project will aim to create an agent, which can then load a gymnasium environment; pre-specify a neural network architecture; and train the network on the agent.

A demonstration of the use of this module is shown below. First, we define our pygment agent and the method of RL being used.

```
import pygment as pm
import gymnasium as gym


# Create a pygment agent (with its method of learning)
agent = pm.create_agent('DQN')
```
Then we define a gymnasium environment and load it into our agent.
```
# Define a gymnasium environment and load it into the agent.
env = gym.make('CartPole-v1', 
                max_episode_steps=500)

agent.load_env(env)
```
Once our agent knows the environment action/observation space, we can specify a neural network architecture and learning parameters.
```
# Define the network architecture and method of learning.
agent.add_network(nodes=[64, 64])
agent.compile(optimizer='adam', learning_rate=0.01)
```
And finally, we can get learning! A print-out of the training losses and rewards will be generated. The module will also
automatically create a folder for the learning episode, intermittently saving checkpoints as specified.
```
agent.train(target_reward=300, 
            save_from=200, 
            save_interval=10,
            episodes=10000,
            gamma=0.99)
```
After training, the trained network can be directly invoked by calling:
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
