import torch
from torch import nn
import torch.functional as F
import numpy as np
import random
import time
from copy import deepcopy
from actions import GreedyEpsilonSelector, calc_loss_prios, calc_loss
from common import wrap_env
from collections import deque
import gymnasium as gym


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions, layers: int, layer_sizes: "list of ints"):
        super().__init__()

        self.net = nn.Sequential()

        for layer in range(layers + 1):
            if layer == 0:
                self.net.append(nn.Linear(input_shape, layer_sizes[layer]))
                self.net.append(nn.ReLU())
            elif layer == layers:
                self.net.append(nn.Linear(layer_sizes[layer - 1], n_actions))
            else:
                self.net.append(nn.Linear(layer_sizes[layer - 1], layer_sizes[layer]))
                self.net.append(nn.ReLU())

    def forward(self, x):
        return self.net(x)

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        for layer in range(len(self.net)):
            if list(map(list, self.net[layer].parameters())):
                self.net[layer].reset_parameters()


class DualNet:
    """
    Wrapper around model to create both a main_net and a target_net
    """

    def __init__(self, model=nn.Sequential()):
        self.main_net = model
        self.target_net = None
        self.action_space = None
        self.observation_space = None
        self.__activations__ = {'relu': nn.ReLU(),
                                'sigmoid': nn.Sigmoid(),
                                'leaky': nn.LeakyReLU()}

    def sync(self):
        self.target_net.load_state_dict(self.main_net.state_dict())


class BaseAgent:
    """
    Abstract Agent interface
    """

    def __init__(self):
        return


class DQNAgent(BaseAgent):
    """
    DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and  converts them into the actions using action_selector
    """

    '''
    def __init__(self, net, optimizer=torch.optim.Adam, epsilon=1, gamma=0.999999, lr=0.001, device="mps"):
        self.net = net
        self.device = device
        self.net.main_net.to(self.device)
        self.net.target_net.to(self.device)
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = lr
        self.optimizer = optimizer(self.net.main_net.parameters(), lr=self.learning_rate)
        self.replay_buffer = deque([], maxlen=10000)
        # self.replay_buffer_priorities = deque([], maxlen=10000)'''

    def __init__(self):
        super().__init__()
        self.net = DualNet(nn.Sequential())
        self.device = 'mps'
        self.epsilon = 1
        self.gamma = 0.999
        self.learning_rate = 0.001
        self.optimizer = None
        self.env = None
        self.__compiled__ = False
        self.replay_buffer = None
        self.__optimizers__ = {'adam': torch.optim.Adam,
                               'sgd': torch.optim.SGD,
                               'rmsprop': torch.optim.RMSprop}

    def reset(self):
        while True:
            reset_input = input('Wipe agent history? Y/N: ')
            if reset_input not in ['y', 'n', 'Y', 'N']:
                print('Please enter a valid value (Y/N)')
                continue
            break

        if reset_input in ['y', 'Y']:
            super().__init__()
            self.net = DualNet(nn.Sequential())
            self.device = 'mps'
            self.epsilon = 1
            self.gamma = 0.999
            self.learning_rate = 0.001
            self.optimizer = None
            self.env = None
            self.__compiled__ = False
            self.replay_buffer = None

    def load_env(self, env, stack_frames=1, reward_clipping=False):
        self.env = wrap_env(env, stack_frames, reward_clipping)
        self.net.action_space = self.env.action_space.n
        self.net.observation_space = self.env.observation_space.shape[0]

    def add_layer(self, neurons, activation):
        if self.__compiled__:
            raise AttributeError('Model is already compiled!')

        if activation not in self.net.__activations__.keys():
            raise KeyError('Please enter a valid activation type: ["relu", "sigmoid", "leaky"]')

        if self.net.action_space is None:
            raise ImportError('Please load a gym environment first!')

        activation = self.net.__activations__[activation]

        if self.net.main_net:
            self.net.main_net.append(nn.Linear(self.net.main_net.net[-1].out_features,
                                               neurons))
            self.net.main_net.append(activation)

        else:
            self.net.main_net.append(nn.Linear(self.net.observation_space,
                                               neurons))
            self.net.main_net.append(activation)

    def compile(self, optimizer, learning_rate=0.001, output_activation='linear'):
        if self.__compiled__:
            raise AttributeError('Model is already compiled!')

        if not self.net.main_net:
            raise AttributeError('Please add a hidden layer before compiling.')

        if optimizer not in self.__optimizers__.keys():
            raise KeyError('Invalid optimizer key -> select one of "adam", "sgd", "rmsprop"')

        self.optimizer = self.__optimizers__[optimizer](self.net.main_net.parameters(),
                                                        lr=learning_rate)
        self.net.main_net.append(nn.Linear(self.net.main_net[-2].out_features,
                                           self.net.action_space))
        self.net.target_net = deepcopy(self.net.main_net)

        self.__compiled__ = True

    def train(self, target_reward, episodes=10000, buffer=10000, target_update=1000):
        if self.env is None:
            raise AttributeError('Please load environment first')
        if not self.net.main_net:
            raise AttributeError('Please add a hidden layer first')
        if not self.__compiled__:
            raise AttributeError('Model must be compiled first')

        self.replay_buffer = deque([], maxlen=buffer)
        self.fill_buffer()

        if not self.full_buffer():
            raise AttributeError('Error with buffer filling')

        total_steps = 0
        max_steps = 500
        total_rewards = []
        total_loss = []
        current_reward = []
        last_reward = []
        current_loss = deque([], maxlen=100)
        start = time.time()

        for episode in range(episodes):
            episode += 1
            num_steps = 0
            last_reward = deepcopy(current_reward)
            current_reward = []

            total_rewards.append(np.array(last_reward).sum())
            total_loss.append(np.array(current_loss).mean())

            prem_done = False
            done = False
            state = env.reset()[0]

            while (not done) and (not prem_done) and (num_steps < max_steps):
                total_steps += 1
                num_steps += 1
                agent.epsilon *= 0.99
                agent.epsilon = max(0.02, agent.epsilon)
                action = agent.action_selector(state)
                next_state, reward, done, prem_done, info = env.step(action)

                loss = agent.process_batch(32)

                agent.buffer_update(Experience(state, action, reward, next_state, done, prem_done))

                state = next_state

                current_loss.append(loss)
                current_reward.append(reward)

                if total_steps % 500 == 0:
                    print('Updating target network...')
                    agent.net.sync()

            if prem_done:
                raise ValueError('Prematurely truncated')

            if np.array(current_reward).sum() >= target_reward:
                end = time.time()
                print(f'Solved in {round((end - start) / 60, 1)} minutes!')
                print(f'Final reward: {np.array(current_reward).sum()}')
                break

    def fill_buffer(self):
        print('Filling buffer...')
        while True:
            state = self.env.reset()[0]
            done = False
            prem_done = False
            max_steps = 1000
            num_steps = 0

            while (not done) and (not prem_done) and (num_steps < max_steps) and (not self.full_buffer()):
                num_steps += 1
                action = np.random.randint(self.env.action_space.n)
                next_state, reward, done, prem_done, info = self.env.step(action)

                self.buffer_update(Experience(state, action, reward, next_state, done, prem_done))

                state = next_state

            if self.full_buffer():
                break

        print('Buffer full.')

    def full_buffer(self):
        return self.replay_buffer.__len__() == self.replay_buffer.maxlen

    def buffer_update(self, sample):  # , sample_weight=None):
        self.replay_buffer.append(sample)

    def action_selector(self, obs):
        return GreedyEpsilonSelector(torch.tensor(obs).to(self.device), self.epsilon, self.net.main_net)

    def process_batch(self, batch_size: int, top_percentile: float = 1.0) -> object:
        batch = self.buffer_sample(batch_size, top_percentile)


        #loss_v, batch_prios = calc_loss_prios(batch, batch_weights, self.device, self.net, self.gamma)
        loss_v = calc_loss(batch, self.device, self.net, self.gamma)

        self.optimizer.zero_grad()
        loss_v.backward()

        self.optimizer.step()
        #self.buffer_update_priorities(batch_indices, batch_prios)

        return loss_v.item()

    def buffer_sample(self, batch_size, top_percentile):
        #normalized_weights = np.array(self.replay_buffer_priorities) ** 0.6
        #normalized_weights = normalized_weights / normalized_weights.sum()

        batch_size = round(batch_size / top_percentile)

        batch_indices = np.random.choice([i for i in range(len(self.replay_buffer))],
                                         size=batch_size,
                                         replace=False)
                                         #p=normalized_weights)
        batch = []
        rewards = []
        for index in batch_indices:
            batch.append(self.replay_buffer[index])
            rewards.append(self.replay_buffer[index].reward)

        if top_percentile < 1:
            filtered_batch_indices = np.argsort(rewards)[:round((1 - top_percentile) * len(rewards))-1:-1]
            new_batch = []
            for new_index in filtered_batch_indices:
                new_batch.append(batch[new_index])

        else:
            new_batch = batch

        # new_batch_indices = batch_indices[filtered_batch_indices]
        # batch_weights = np.array(self.replay_buffer_priorities)[batch_indices]
        return new_batch  # , new_batch_indices , batch_weights


class Experience:
    def __init__(self, state, action, reward, next_state, done, prem_done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.prem_done = prem_done
