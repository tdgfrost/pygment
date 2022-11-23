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
from .common import wrap_env
from collections import deque
import ray


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
            self._optimizer = None
            self._learning_rate = None
            self.__output_activation = None
            self._compiled = False
            self._gamma = None
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


class PolicyGradient(BaseAgent):
    """
    PolicyGradient is a neural network based on REINFORCE
    """

    def __init__(self):
        super().__init__()
        self.net = PolicyGradientNet()
        self.predict_net = None

    def add_network(self, nodes: list):
        if (not isinstance(nodes, list)) or (not np.issubdtype(np.array(nodes).dtype, np.integer)):
            raise TypeError('Node values must be entered as integers within a list')

        self.net.add_layers(len(nodes), nodes, self.observation_space, self.action_space)
        self.predict_net = self.net.cpu()
        self.net.has_net = True


    def compile(self, optimizer, learning_rate=0.001):
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        super().compile_check()

        self.net.to(self.device)
        self.predict_net.to('cpu')
        self.optimizer = self._optimizers[self._optimizer](self.net.parameters(),
                                                           lr=self._learning_rate)

        self._compiled = True


    def train(self, target_reward=None, episodes=10000, ep_update=4, gamma=0.999, max_steps=None):
        self.method_check(env_loaded=True, net_exists=True, compiled=True)
        self._gamma = gamma
        if max_steps is None:
            max_steps = self._max_steps if self._max_steps is not None else 10000


        total_rewards = deque([], maxlen=100)
        total_loss = deque([], maxlen=100)
        for episode in range(episodes // ep_update):

            @ray.remote
            def env_run():
                state_record = []
                reward_record = []
                action_record = []
                cum_reward = []

                state = self.env.reset()[0]
                state_record.append(state)
                for _ in range(max_steps):
                  with torch.no_grad():
                      action, _, _ = self.predict_net.forward(state, device='cpu')
                      action_record.append(action)
                  state, reward, done, _, _ = self.env.step(action)

                  reward_record.append(reward)

                  if done:
                    break
                  else:
                    state_record.append(state)

                total_reward = np.array(reward_record).sum()

                cum_r = calc_cum_rewards(reward_record, self._gamma)
                for r in cum_r:
                  cum_reward.append(r)

                return state_record, action_record, cum_reward, total_reward

            state_record, action_record, cum_reward, total_reward = zip(*ray.get([env_run.remote() for _ in range(ep_update)]))

            [total_rewards.append(r) for r in total_reward]
            if (target_reward is not None) & (len(total_rewards) == 100):
                if np.array(total_rewards).mean() > target_reward:
                    print(f'Solved at target {target_reward}!')
                    break

            state_records = [state for record in state_record for state in record]
            cum_rewards = [cum_r for record in cum_reward for cum_r in record]
            action_records = [action for record in action_record for action in record]
            action_probs_records = []
            action_logprobs_records = []
            for state in state_records:
                _, action_probs, action_logprobs = self.net.cpu().forward(state, device='cpu')
                action_probs_records.append(action_probs)
                action_logprobs_records.append(action_logprobs)

            # calculate loss
            self.optimizer.zero_grad()
            loss = calc_loss_policy(cum_rewards, action_records, action_logprobs_records, device='mps')
            loss.backward(retain_graph=True)
            # Now calculate the entropy loss

            entropy_loss = calc_entropy_loss_policy(action_probs_records, action_logprobs_records, device='mps')
            entropy_loss.backward()
            # update the model and predict_model
            self.optimizer.step()
            self.predict_net.load_state_dict(self.net.state_dict())

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


    def train(self, episodes=10000, gamma=0.999, max_steps=None):
        self.method_check(env_loaded=True, net_exists=True, compiled=True)
        self._gamma = gamma
        if max_steps is None:
            max_steps = self._max_steps if self.max_steps is not None else 10000

        for episode in range(episodes):
            state = self.env.reset()[0]
            for num_step in range(max_steps):
                action = self.net(state)
                state, reward, done, _, _ = self.env.step(action)
        # UNFINISHED



    def action_selector(self):
        pass


    def reset(self):
        reset_done = super().reset()

        if reset_done:
            self.net = ActorCriticNet()



class DQNAgent(BaseAgent):
    """
    DQNAgent is a DQN agent which calculates Q values
    from the observations and  converts them into the actions using action_selector
    """

    def __init__(self):
        super().__init__()
        self.net = DualNet(nn.Sequential())
        self.replay_buffer = None
        self._tau = None
        self._batch_size = None


    def reset(self):
        reset_done = super().reset()

        if reset_done:
            self.net = DualNet(nn.Sequential())
            self.replay_buffer = None
            self._tau = None
            self._batch_size = None


    def add_layer(self, neurons, activation):
        if self.env is None:
            raise ImportError('Please load a gym environment first!')

        if self._compiled:
            raise AttributeError('Model is already compiled!')

        if activation not in self.net.activations.keys():
            raise KeyError('Please enter a valid activation type: ["relu", "sigmoid", "leaky"]')

        activation = self.net.activations[activation]()

        if self.net.main_net:
            self.net.main_net.append(nn.Linear(self.net.main_net[-2].out_features,
                                               neurons))
            self.net.main_net.append(activation)

        else:
            self.net.observation_space = self.observation_space
            self.net.action_space = self.action_space
            self.net.main_net.append(nn.Linear(self.net.observation_space,
                                               neurons))
            self.net.main_net.append(activation)

        self.net.has_net = True


    def compile(self, optimizer, learning_rate=0.001, output_activation='linear'):
        self._optimizer = optimizer
        self.__output_activation = output_activation
        self.compile_check()
        self.optimizer = self._optimizers[optimizer](self.net.main_net.parameters(),
                                                        lr=learning_rate)
        self.output_activation = output_activation

        self.net.main_net.append(nn.Linear(self.net.main_net[-2].out_features,
                                           self.net.action_space))
        if self.net.output_activations[output_activation] is not None:
            self.net.main_net.append(self.net.output_activations[output_activation]())

        self.net.main_net.to(self.device)
        self.net.target_net = deepcopy(self.net.main_net)
        self._compiled = True


    def train(self, target_reward, episodes=10000, batch_size=64, buffer=10000, gamma=0.999999,
              epsilon=1, tau=0.001, decay_rate=0.999, min_epsilon=0.02, max_steps=None):
        self.method_check(env_loaded=True, net_exists=True, compiled=True)
        self.replay_buffer = deque([], maxlen=buffer)
        self._gamma = gamma
        self._epsilon = epsilon
        self._tau = tau
        self._eps_decay_rate = decay_rate
        self._min_epsilon = min_epsilon
        self._batch_size = batch_size
        if max_steps is None:
            max_steps = self._max_steps if self.max_steps is not None else 10000

        self.fill_buffer(max_steps)

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

            for num_steps in range(max_steps):
                total_steps += 1
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

                if done or prem_done:
                    break

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


    def fill_buffer(self, max_steps):
        self.method_check(env_loaded=True, net_exists=True, compiled=True)
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


    def action_selector(self, obs, no_epsilon=False):
        self.method_check(env_loaded=True, net_exists=True, compiled=True)
        if no_epsilon:
          return GreedyEpsilonSelector(torch.tensor(obs).to(self.device), 0, self.net.main_net)

        return GreedyEpsilonSelector(torch.tensor(obs).to(self.device), self._epsilon, self.net.main_net)


    def process_batch(self, batch_size: int, top_percentile: float = 1.0) -> object:
        self.method_check(env_loaded=True, net_exists=True, compiled=True)
        batch = self.buffer_sample(batch_size, top_percentile)
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


    def buffer_sample(self, batch_size, top_percentile):
        self.method_check(env_loaded=True, net_exists=True, compiled=True)
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


    def compile_check(self):
        super().compile_check()

        if self.__output_activation not in self.net.output_activations.keys():
            raise KeyError('Invalid output_activation key -> select one of "linear", "sigmoid", "softmax"')


class Experience:
    def __init__(self, state, action, reward, next_state, done, prem_done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.prem_done = prem_done
