import torch
from torch import nn
import numpy as np
import random
import time
from copy import deepcopy
from .actions import GreedyEpsilonSelector, calc_loss
from .net import DualNet
from .common import wrap_env
from collections import deque


class BaseAgent:
    """
    Abstract Agent interface
    """

    def __init__(self):
        self.device = 'mps'
        self.optimizer = None
        self.env = None
        self.action_space = None
        self.observation_space = None
        self.output_activation = None
        self._compiled = False
        self._epsilon = None
        self._eps_decay_rate = None
        self._min_epsilon = None
        self._max_steps = None
        self._optimizers = {'adam': torch.optim.Adam,
                               'sgd': torch.optim.SGD,
                               'rmsprop': torch.optim.RMSprop}
        return


    def reset(self):
        while True:
            reset_input = input('Wipe agent history? Y/N: ')
            if reset_input not in ['y', 'n', 'Y', 'N']:
                print('Please enter a valid value (Y/N)')
                continue
            break

        if reset_input in ['y', 'Y']:
            self.device = 'mps'
            self.optimizer = None
            self.env = None
            self.action_space = None
            self.observation_space = None
            self.output_activation = None
            self._compiled = False
            self._epsilon = None
            self._eps_decay_rate = None
            self._min_epsilon = None
            self._max_steps = None
            self._optimizers = {'adam': torch.optim.Adam,
                                   'sgd': torch.optim.SGD,
                                   'rmsprop': torch.optim.RMSprop}
            reset_done = True
        else:
            reset_done = False

        return reset_done

    def load_env(self, env, stack_frames=1, reward_clipping=False):
        self.env = wrap_env(env, stack_frames, reward_clipping)
        self.action_space = self.env.action_space.n
        self.observation_space = self.env.observation_space.shape[0]
        try:
            self._max_steps = self.env._max_episode_steps
        except:
            pass


class ActorCritic(BaseAgent):
    """
    ActorCritic uses the actor-critic neural network to solve environments
    """

    def __init__(self):
        super().__init__()



class DQNAgent(BaseAgent):
    """
    DQNAgent is a DQN agent which calculates Q values
    from the observations and  converts them into the actions using action_selector
    """

    def __init__(self):
        super().__init__()
        self.net = DualNet(nn.Sequential())
        self.replay_buffer = None
        self._gamma = None
        self._tau = None
        self._batch_size = None


    def reset(self):
        reset_done = super().reset()

        if reset_done:
            self.net = DualNet(nn.Sequential())
            self.replay_buffer = None
            self._gamma = None
            self._tau = None
            self._batch_size = None


    def add_layer(self, neurons, activation):
        if self.env is None:
            raise ImportError('Please load a gym environment first!')

        if self._compiled:
            raise AttributeError('Model is already compiled!')

        if activation not in self.net.activations.keys():
            raise KeyError('Please enter a valid activation type: ["relu", "sigmoid", "leaky"]')

        activation = self.net.activations[activation]

        if self.net.main_net:
            self.net.main_net.append(nn.Linear(self.net.main_net[-2].out_features,
                                               neurons))
            self.net.main_net.append(activation)

        else:
            self.net.main_net.append(nn.Linear(self.net.observation_space,
                                               neurons))
            self.net.main_net.append(activation)


    def compile(self, optimizer, learning_rate=0.001, output_activation='linear'):
        self.compile_check(optimizer, output_activation)
        self.optimizer = self._optimizers[optimizer](self.net.main_net.parameters(),
                                                        lr=learning_rate)
        self.output_activation = output_activation

        self.net.main_net.append(nn.Linear(self.net.main_net[-2].out_features,
                                           self.net.action_space))
        if self.net.output_activations[output_activation] is not None:
            self.net.main_net.append(self.net.output_activations[output_activation])

        self.net.main_net.to(self.device)
        self.net.target_net = deepcopy(self.net.main_net)
        self._compiled = True


    def train(self, target_reward, episodes=10000, batch_size=64, buffer=10000, gamma=0.999999,
              epsilon=1, tau=0.001, decay_rate=0.999, min_epsilon=0.02, max_steps=None):
        self.method_check()
        self.replay_buffer = deque([], maxlen=buffer)
        self._gamma = gamma
        self._epsilon = epsilon
        self._tau = tau
        self._eps_decay_rate = decay_rate
        self._min_epsilon = min_epsilon
        self._batch_size = batch_size
        if max_steps is None:
            max_steps = self._max_steps

        self.fill_buffer()

        if not self.full_buffer():
            raise AttributeError('Error with buffer filling')

        total_steps = 0
        total_rewards = []
        total_loss = []
        current_reward = []
        current_loss = deque([], maxlen=100)
        total_start = time.time()
        local_start = time.time()

        for episode in range(episodes):
            if episode % 50 == 0:
                print(f'Episodes completed: {episode}')
            num_steps = 0
            last_reward = deepcopy(current_reward)
            current_reward = []

            total_rewards.append(np.array(last_reward).sum())
            total_loss.append(np.array(current_loss).mean())

            prem_done = False
            done = False
            state = self.env.reset()[0]

            while (not done) and (not prem_done) and (num_steps < max_steps):
                total_steps += 1
                num_steps += 1
                self._epsilon *= self._eps_decay_rate
                self._epsilon = max(self._min_epsilon, self._epsilon)
                action = self.action_selector(state)
                next_state, reward, done, prem_done, info = self.env.step(action)

                loss = self.process_batch(self._batch_size)

                self.buffer_update(Experience(state, action, reward, next_state, done, prem_done))

                state = next_state

                current_loss.append(loss)
                current_reward.append(reward)

                self.net.sync(self._tau)

                if (time.time() - local_start >= 2) & (len(last_reward) > 0):
                    print(f'Reward: {np.array(last_reward).sum()}, Loss: {np.array(current_loss).mean()}')
                    last_reward = []
                    local_start = time.time()

            if len(total_rewards) >= 100:
                if np.array(total_rewards[-100:]).mean() >= target_reward:
                    total_end = time.time()
                    duration = total_end - total_start
                    if duration < 300:
                        print(f'Solved in {round((total_end - total_start), 0)} seconds!')
                    elif duration < 3600:
                        print(f'Solved in {round((total_end - total_start) / 60, 1)} minutes!')
                    else:
                        print(f'Solved in {round((total_end - total_start) / 3600, 1)} hours!')

                    print(f'Final reward: {np.array(current_reward).sum()}')
                    break


    def fill_buffer(self):
        self.method_check()
        print('Filling buffer...')
        while True:
            state = self.env.reset()[0]
            done = False
            prem_done = False
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
        self.method_check()
        return self.replay_buffer.__len__() == self.replay_buffer.maxlen


    def buffer_update(self, sample):
        self.method_check()
        self.replay_buffer.append(sample)


    def action_selector(self, obs, no_epsilon=False):
        self.method_check()
        if no_epsilon:
          return GreedyEpsilonSelector(torch.tensor(obs).to(self.device), 0, self.net.main_net)

        return GreedyEpsilonSelector(torch.tensor(obs).to(self.device), self._epsilon, self.net.main_net)


    def process_batch(self, batch_size: int, top_percentile: float = 1.0) -> object:
        self.method_check()
        batch = self.buffer_sample(batch_size, top_percentile)
        loss_v = calc_loss(batch, self.device, self.net, self._gamma)

        self.optimizer.zero_grad()
        loss_v.backward()
        self.optimizer.step()

        return loss_v.item()


    def buffer_sample(self, batch_size, top_percentile):
        self.method_check()
        batch_size = round(batch_size / top_percentile)

        batch_indices = np.random.choice([i for i in range(len(self.replay_buffer))],
                                         size=batch_size,
                                         replace=False)
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

        return new_batch


    def method_check(self):
        if self.env is None:
            raise AttributeError('Please load environment first')
        if not self.net.main_net:
            raise AttributeError('Please add a hidden layer first')
        if not self._compiled:
            raise AttributeError('Model must be compiled first')


    def compile_check(self, optimizer, output_activation):
      if self._compiled:
        raise AttributeError('Model is already compiled!')

      if not self.net.main_net:
        raise AttributeError('Please add a hidden layer before compiling.')

      if optimizer not in self._optimizers.keys():
        raise KeyError('Invalid optimizer key -> select one of "adam", "sgd", "rmsprop"')

      if output_activation not in self.net.output_activations.keys():
        raise KeyError('Invalid output_activation key -> select one of "linear", "sigmoid", "softmax"')


class Experience:
    def __init__(self, state, action, reward, next_state, done, prem_done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.prem_done = prem_done
