import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import random
import time
from copy import deepcopy
from .actions import GreedyEpsilonSelector, calc_loss_batch, calc_loss_policy, calc_cum_rewards, \
  calc_entropy_loss_policy
from .net import DualNet, ActorCriticNet, PolicyGradientNet
from .env import wrap_env
from collections import deque
import ray


class Experience:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


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
        self._optimizer = None
        self._learning_rate = None
        self.__output_activation = None
        self._compiled = False
        self._gamma = None
        self._epsilon = None
        self._eps_decay_rate = None
        self._min_epsilon = None
        self._regularisation = None
        self._optimizers = {'adam': torch.optim.Adam,
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
            self.device = 'mps'
            self.optimizer = None
            self.env = None
            self.action_space = None
            self.observation_space = None
            self.output_activation = None
            self._optimizer = None
            self._learning_rate = None
            self.__output_activation = None
            self._compiled = False
            self._gamma = None
            self._epsilon = None
            self._eps_decay_rate = None
            self._min_epsilon = None
            self._regularisation = None
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


    def compile_check(self):
        if self._compiled:
            raise AttributeError('Model is already compiled!')
        if self._optimizer not in self._optimizers.keys():
            raise KeyError('Invalid optimizer key -> select one of "adam", "sgd", "rmsprop"')
        if not self.net.has_net:
            raise AttributeError('Please add a neural network before compiling.')


    def method_check(self, env_loaded, net_exists, compiled):
        if (self.env is None) and env_loaded:
            raise AttributeError('Please load environment first')
        if (not self.net.has_net) and net_exists:
            raise AttributeError('Please add a hidden layer first')
        if (not self._compiled) and compiled:
            raise AttributeError('Model must be compiled first')


class DQNAgent(BaseAgent):
    """
    DQNAgent is a DQN agent which calculates Q values
    from the observations and  converts them into the actions using action_selector
    """

    def __init__(self):
        super().__init__()
        self.net = DualNet()
        self.replay_buffer = None
        self._tau = None
        self._batch_size = None


    def reset(self):
        reset_done = super().reset()

        if reset_done:
            self.net = DualNet()
            self.replay_buffer = None
            self._tau = None
            self._batch_size = None


    def add_network(self, nodes: list):
        if (not isinstance(nodes, list)) or (not np.issubdtype(np.array(nodes).dtype, np.integer)):
            raise TypeError('Node values must be entered as integers within a list')

        if self._compiled:
            raise AttributeError('Model is already compiled!')

        if self.env is None:
            raise ImportError('Please load a gym environment first!')

        self.net.add_layers(nodes, self.observation_space, self.action_space)
        self.net.has_net = True


    def compile(self, optimizer, learning_rate=0.001, weight_decay=1e-5, clip=1.0, lower_clip=None, upper_clip=None):
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._regularisation = weight_decay
        super().compile_check()

        self.net.main_net.to(self.device)
        self.net.target_net = deepcopy(self.net.main_net)
        self.net.target_net.to(self.device)

        self.optimizer = self._optimizers[self._optimizer](self.net.main_net.parameters(),
                                                           lr=self._learning_rate,
                                                           weight_decay=self._regularisation)

        for p in self.net.main_net.parameters():
            p.register_hook(lambda grad: torch.clamp(grad,
                                                     lower_clip if lower_clip is not None else -clip,
                                                     upper_clip if upper_clip is not None else clip))

        self._compiled = True


    def train(self, target_reward, episodes=10000, batch_size=64, gamma=0.999, buffer=10000,
              epsilon=1, tau=0.001, decay_rate=0.999, min_epsilon=0.02):
        self.method_check(env_loaded=True, net_exists=True, compiled=True)
        self._buffer_size = buffer
        self._gamma = gamma
        self._epsilon = epsilon
        self._tau = tau
        self._eps_decay_rate = decay_rate
        self._min_epsilon = min_epsilon
        self._batch_size = batch_size

        self.replay_buffer = deque([], maxlen=self._buffer_size)
        self.fill_buffer()

        if not self.full_buffer():
            raise AttributeError('Error with buffer filling')

        total_rewards = deque([], maxlen=100)
        total_loss = deque([], maxlen=100)

        for episode in range(1, episodes+1):
            if episode % 5 == 0:
                print(f'Episodes completed: {episode}, Loss: {np.array(total_loss).mean()}, Reward: {np.array(total_rewards).mean()}')

            prem_done = False
            done = False
            state = self.env.reset()[0]
            current_loss = []
            current_reward = []

            while not done and not prem_done:
                self._epsilon *= self._eps_decay_rate
                self._epsilon = max(self._min_epsilon, self._epsilon)
                action = self.action_selector(state)
                next_state, reward, done, prem_done, _ = self.env.step(action)

                loss = self.process_batch(self._batch_size)

                self.buffer_update(Experience(state, action, reward, next_state, done or prem_done))

                state = next_state

                current_loss.append(loss)
                current_reward.append(reward)

                self.net.sync(self._tau)

            total_rewards.append(np.array(current_reward).sum())
            total_loss += deque(current_loss)

            if len(total_rewards) == 100:
                if np.array(total_rewards).mean() >= target_reward:
                    print(f'Final reward: {np.array(total_rewards).sum()}')
                    break


    def fill_buffer(self):
        self.method_check(env_loaded=True, net_exists=True, compiled=True)
        print('Filling buffer...')

        @ray.remote
        def distributed_buffer_fill(env):
            sample_record = []

            state = env.reset()[0]
            done = False
            prem_done = False
            while not done and not prem_done:

                with torch.no_grad():
                    action = np.random.randint(env.action_space.n)

                next_state, reward, done, prem_done, _ = env.step(action)

                sample = Experience(state, action, reward, next_state, done or prem_done)

                sample_record.append(sample)

                state = next_state

            return sample_record


        while True:
            sample_record = zip(*ray.get([distributed_buffer_fill.remote(deepcopy(self.env)) for _ in range(100)]))

            for sample in sample_record:
                self.replay_buffer += sample

            if self.full_buffer():
                break

        print('Buffer full.')


    def action_selector(self, obs, no_epsilon=False):
        self.method_check(env_loaded=True, net_exists=True, compiled=True)
        if no_epsilon:
          return GreedyEpsilonSelector(torch.tensor(obs).to(self.device), 0, self.net)

        return GreedyEpsilonSelector(torch.tensor(obs).to(self.device), self._epsilon, self.net)


    def process_batch(self, batch_size: int) -> object:
        self.method_check(env_loaded=True, net_exists=True, compiled=True)
        batch = self.buffer_sample(batch_size)

        loss_v = calc_loss_batch(batch, self.device, self.net, self._gamma)

        self.optimizer.zero_grad()
        loss_v.backward()
        self.optimizer.step()

        return loss_v.item()


    def full_buffer(self):
        self.method_check(env_loaded=True, net_exists=True, compiled=True)
        return self.replay_buffer.__len__() == self.replay_buffer.maxlen


    def buffer_update(self, sample):
        self.method_check(env_loaded=True, net_exists=True, compiled=True)
        self.replay_buffer.append(sample)


    def buffer_sample(self, batch_size):
        self.method_check(env_loaded=True, net_exists=True, compiled=True)

        batch_indices = np.random.choice([i for i in range(len(self.replay_buffer))],
                                         size=batch_size,
                                         replace=False)
        batch = []
        rewards = []
        for index in batch_indices:
            batch.append(self.replay_buffer[index])

        return batch


class PolicyGradient(BaseAgent):
    """
    PolicyGradient is a neural network based on REINFORCE
    """

    def __init__(self):
        super().__init__()
        self.net = PolicyGradientNet()


    def reset(self):
        reset_done = super().reset()

        if reset_done:
            self.net = PolicyGradientNet()


    def add_network(self, nodes: list):
        if (not isinstance(nodes, list)) or (not np.issubdtype(np.array(nodes).dtype, np.integer)):
            raise TypeError('Node values must be entered as integers within a list')

        if self._compiled:
            raise AttributeError('Model is already compiled!')

        if self.env is None:
            raise ImportError('Please load a gym environment first!')

        self.net.add_layers(nodes, self.observation_space, self.action_space)
        self.net.has_net = True


    def compile(self, optimizer, learning_rate=0.001, weight_decay=1e-5, clip=1.0, lower_clip=None, upper_clip=None):
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._regularisation = weight_decay
        super().compile_check()

        self.net.to(self.device)
        self.optimizer = self._optimizers[self._optimizer](self.net.parameters(),
                                                           lr=self._learning_rate,
                                                           weight_decay=self._regularisation)

        for p in self.net.parameters():
            p.register_hook(lambda grad: torch.clamp(grad,
                                                     lower_clip if lower_clip is not None else -clip,
                                                     upper_clip if upper_clip is not None else clip))

        self._compiled = True


    def train(self, target_reward=None, episodes=10000, ep_update=4, gamma=0.999):
        self.method_check(env_loaded=True, net_exists=True, compiled=True)
        self._gamma = gamma


        total_rewards = deque([], maxlen=100)
        total_loss = deque([], maxlen=100)

        @ray.remote
        def env_run(predict_net):
          state_record = []
          reward_record = []
          action_record = []
          cum_reward = []

          state = self.env.reset()[0]
          done = False
          prem_done = False
          while not done and not prem_done:
              state_record.append(torch.tensor(state))

              with torch.no_grad():
                  action, _, _ = predict_net.forward(state, device='cpu')
                  action_record.append(action.item())

              state, reward, done, prem_done, _ = self.env.step(action.item())

              reward_record.append(reward)

          total_reward = np.array(reward_record).sum()

          cum_r = calc_cum_rewards(reward_record, self._gamma)
          for r in cum_r:
            cum_reward.append(r)

          return state_record, action_record, cum_reward, total_reward

        for episode in range(episodes // ep_update):

            state_record, action_record, cum_reward, total_reward = zip(*ray.get([env_run.remote(self.net.cpu()) for _ in range(ep_update)]))

            [total_rewards.append(r) for r in total_reward]
            if (target_reward is not None) & (len(total_rewards) == 100):
                if np.array(total_rewards).mean() > target_reward:
                    print(f'Solved at target {target_reward}!')
                    break

            state_records = []
            for record in state_record:
                state_records += record
            state_records = torch.stack(state_records).to('mps')
            cum_rewards = [cum_r for record in cum_reward for cum_r in record]
            action_records = [action for record in action_record for action in record]
            self.net.to('mps')
            _, action_probs_records, action_logprobs_records = self.net(state_records)

            # calculate loss
            self.optimizer.zero_grad()
            loss = calc_loss_policy(cum_rewards, action_records, action_logprobs_records, device='mps')
            loss.backward(retain_graph=True)
            # Now calculate the entropy loss

            entropy_loss = calc_entropy_loss_policy(action_probs_records, action_logprobs_records, device='mps')
            entropy_loss.backward()
            # update the model and predict_model
            self.optimizer.step()
            total_loss.append(loss.item())

            print(f'Episodes: {episode * ep_update}, Loss {np.array(total_loss).mean()}, Mean Reward: {np.array(total_rewards).mean()}')


                # At some point, change all these lists to the Experience class
                # (complete with a .reset() or .clear() function)

    def action_selector(self, obs):
        self.method_check(env_loaded=True, net_exists=True, compiled=True)

        return self.net(obs)[0]


class ActorCritic(BaseAgent):
    """
    ActorCritic uses the actor-critic neural network to solve environments
    """

    def __init__(self):
        super().__init__()
        self.net = ActorCriticNet()


    def add_network(self, layers=1, nodes=None):
        if nodes is None:
            nodes = [128]

        if (not isinstance(layers, int)) or (layers == 0):
            raise ValueError('Layers must be a non-zero integer')
        if (not isinstance(nodes, list)) or (not np.issubdtype(np.array(nodes).dtype, np.integer)):
            raise TypeError('Node values must be entered as integers within a list')
        if len(nodes) != layers:
            raise AttributeError('Layers and nodes must match')

        self.net.add_layers(layers, nodes, self.observation_space, self.action_space)
        self.net.has_net = True


    def compile(self, optimizer, learning_rate=0.001):
        self._optimizer = optimizer
        super().compile_check()

        self.optimizer = self._optimizers[optimizer](self.net.parameters(),
                                                     lr=learning_rate)

        self._compiled = True


    def train(self, episodes=10000, gamma=0.999):
        self.method_check(env_loaded=True, net_exists=True, compiled=True)
        self._gamma = gamma

        for episode in range(episodes):
            state = self.env.reset()[0]

            action = self.net(state)
            state, reward, done, _, _ = self.env.step(action)
        # UNFINISHED



    def action_selector(self):
        pass


    def reset(self):
        reset_done = super().reset()

        if reset_done:
            self.net = ActorCriticNet()

