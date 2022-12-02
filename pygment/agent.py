import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import random
import time
import datetime as dt
import os
from copy import deepcopy
from .actions import GreedyEpsilonSelector, calc_loss_batch, calc_loss_policy, calc_cum_rewards, \
  calc_entropy_loss_policy, calc_loss_actor_critic
from .net import DualNet, ActorCriticNet, PolicyGradientNet
from .env import wrap_env
from collections import deque
import ray

class Experience:
  '''
  The Experience class stores values from a training episode (values from the environment
  and values from the network).
  '''
  def __init__(self, state=None, action=None, reward=None, next_state=None, done=None,
                 action_probs=None, action_logprobs=None, state_value=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.action_probs = action_probs
        self.action_logprobs = action_logprobs
        self.state_value = state_value


class BaseAgent:
    """
    Base Agent stores universal attributes and super() methods across the agents.
    """

    def __init__(self):
        self.device = 'mps'
        self.optimizer = None
        self.env = None
        self.action_space = None
        self.observation_space = None
        self.output_activation = None
        self.current_best_reward = -10**100
        now = dt.datetime.now()
        self._path = f'./{now.year}_{now.month}_{now.day:02}_{now.hour:02}_{now.minute:02}_{now.second:02}'
        self._optimizer = None
        self._learning_rate = None
        self.__output_activation = None
        self._compiled = False
        self._gamma = None
        self._epsilon = None
        self._eps_decay_rate = None
        self._min_epsilon = None
        self._regularisation = None
        self._screen_files = True
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
            self.current_best_reward = -10**100
            now = dt.datetime.now()
            self._path = f'./{now.year}_{now.month}_{now.day:02}_{now.hour:02}_{now.minute:02}_{now.second:02}/'
            self._optimizer = None
            self._learning_rate = None
            self.__output_activation = None
            self._compiled = False
            self._gamma = None
            self._epsilon = None
            self._eps_decay_rate = None
            self._min_epsilon = None
            self._regularisation = None
            self._screen_files = True
            self._optimizers = {'adam': torch.optim.Adam,
                                'sgd': torch.optim.SGD,
                                'rmsprop': torch.optim.RMSprop}
            reset_done = True
        else:
            reset_done = False

        return reset_done

    '''
    load_env stores the environment (and its action/observation space) within the agent, 
    but also offers the option for frame stacking and reward clipping.
    '''
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

    def network_check(self, nodes):
      if (not isinstance(nodes, list)) or (not np.issubdtype(np.array(nodes).dtype, np.integer)):
        raise TypeError('Node values must be entered as integers within a list')
      if self._compiled:
        raise AttributeError('Model is already compiled!')
      if self.env is None:
        raise ImportError('Please load a gym environment first!')

    def method_check(self, env_loaded, net_exists, compiled):
        if (self.env is None) and env_loaded:
            raise AttributeError('Please load environment first')
        if (not self.net.has_net) and net_exists:
            raise AttributeError('Please add a hidden layer first')
        if (not self._compiled) and compiled:
            raise AttributeError('Model must be compiled first')

    '''
    The train super() method ensures that the optimizer is correctly synced to the model,
    and creates a directory in which the model checkpoints and any animations can be saved.
    '''
    def train(self, gamma):
      self.method_check(env_loaded=True, net_exists=True, compiled=True)
      self._gamma = gamma

      if not os.path.isdir(self._path):
        os.mkdir(self._path)

      self.optimizer = self._optimizers[self._optimizer](self.net.parameters(),
                                                         lr=self._learning_rate,
                                                         weight_decay=self._regularisation)

    '''
    save_model does several important steps/checks prior to model saving, which are detailed below.
    '''
    def save_model(self, average_reward, save_from, save_interval, best=False):

      # Check that the current model performance exceeds the 'save_from' threshold,
      # and that the current model is better than the previous 'best' model.
      if (average_reward >= save_from) & ((average_reward // save_interval * save_interval) >
                                          (self.current_best_reward // save_interval * save_interval)):

        # It may be the case that a previous model was loaded and training was continued. If this hasn't been
        # checked yet, screen the previous checkpoints to check whether any were saved as 'model_best_',
        # and clarify the score of that checkpoint.
        if self._screen_files:
          no_best = True
          for file in os.listdir(self._path):
            if 'model_best_' in file:
              no_best = False
              prev_score = file.split('model_best_')[1].split('.pt')[0]

              # If the score of this checkpoint is below the current model, remove the 'model_best_' from that
              # checkpoint.
              if int(float(prev_score)) < (average_reward // save_interval * save_interval):
                os.rename(self._path+f'/{file}', self._path+f'/model_{int(float(prev_score))}.pt')
                self._screen_files = False
                break

          if no_best:
            self._screen_files = False

        # Define the directory to save the model in, and the name of the file, which follows the template of
        # 'model_(score)' initially, and 'model_best_(score)' at the terminus.
        current_model_path = self._path + f'/model_best_{int(average_reward // save_interval * save_interval)}.pt' if best \
          else self._path + f'/model_{int(average_reward // save_interval * save_interval)}.pt'

        # For some reason there are issues with saving and re-training the model under mps, so (for the time being)
        # we create a deepcopy of our model to get around this.
        current_model = deepcopy(self.net)
        torch.save(current_model, current_model_path)


    def load_model(self):
      invalid_path = ''
      while True:
        path = input(invalid_path + 'Please enter model path, or press Q/q to exist: ')
        if path == 'Q' or path == 'q':
          return
        if os.path.isfile(path):
          break
        invalid_path = 'Invalid path - '
      self.net = torch.load(path)
      if self._compiled:
        self._compiled = False
        self._optimizer = None
        self._learning_rate = None
        self._regularisation = None
        print('Model loaded - recompile agent please')

      self._path = os.path.dirname(path)


class DQNAgent(BaseAgent):
    """
    DQNAgent is a Double DQN agent which learns from its environment using a deep Q network (with both a
    current network and an intermittently synchronised target network).
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

    '''
    Feed the desired number of nodes and the associated input/output 
    features to the DualNet network-generating method.
    '''
    def add_network(self, nodes: list):
        super().network_check(nodes)

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
            buffer_record = []

            state = env.reset()[0]
            done = False
            prem_done = False
            while not done and not prem_done:

                with torch.no_grad():
                    action = np.random.randint(env.action_space.n)

                next_state, reward, done, prem_done, _ = env.step(action)

                sample = Experience(state, action, reward, next_state, done or prem_done)

                buffer_record.append(sample)

                state = next_state

            return buffer_record


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
        super().network_check(nodes)

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
          cum_reward = []
          buffer_record = []

          state = self.env.reset()[0]
          done = False
          prem_done = False
          while not done and not prem_done:
              with torch.no_grad():
                  action, _, _ = predict_net.forward(state, device='cpu')

              next_state, reward, done, prem_done, _ = self.env.step(action.item())

              buffer_record.append(Experience(state, action.item(), reward, next_state, done))

              state = next_state

          reward_record = [exp.reward for exp in buffer_record]
          total_reward = np.array(reward_record).sum()
          cum_reward = calc_cum_rewards(reward_record, self._gamma)

          return buffer_record, cum_reward, total_reward

        for episode in range(episodes // ep_update):

            buffer_record, cum_reward, total_reward = zip(
              *ray.get([env_run.remote(self.net.cpu()) for _ in range(ep_update)]))

            [total_rewards.append(r) for r in total_reward]
            if (target_reward is not None) & (len(total_rewards) == 100):
                if np.array(total_rewards).mean() > target_reward:
                    print(f'Solved at target {target_reward}!')
                    break

            state_records, action_records = zip(*[(exp.state, exp.action) for buffer in buffer_record for exp in buffer])
            cum_rewards = [cum_r for record in cum_reward for cum_r in record]
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


    def reset(self):
        reset_done = super().reset()

        if reset_done:
            self.net = ActorCriticNet()


    def add_network(self, nodes: list):
        super().network_check(nodes)

        self.net.add_layers(nodes, self.observation_space, self.action_space)
        self.net.has_net = True

    def compile(self, optimizer, learning_rate=0.001, weight_decay=1e-5, clip=1, lower_clip=None, upper_clip=None):
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._regularisation = weight_decay
        super().compile_check()

        self.net.to(self.device)

        for p in self.net.parameters():
            p.register_hook(lambda grad: torch.clamp(grad,
                                                     lower_clip if lower_clip is not None else -clip,
                                                     upper_clip if upper_clip is not None else clip))

        self._compiled = True


    def train(self, target_reward, save_from, save_interval=10, episodes=10000, parallel_envs=32, gamma=0.999):
        super().train(gamma)

        total_rewards = deque([], maxlen=100)
        total_loss = deque([], maxlen=100)

        @ray.remote
        def env_run(predict_net):
          done = False
          prem_done = False
          state = self.env.reset()[0]
          ep_record = []
          while not done and not prem_done:
            with torch.no_grad():
              action, _, _, state_value = predict_net.forward(state, device='cpu')

            next_state, reward, done, prem_done, _ = self.env.step(action.item())

            ep_record.append(Experience(state=state,
                                        action=action,
                                        reward=reward))

            state = next_state

          cum_rewards = calc_cum_rewards([exp.reward for exp in ep_record], self._gamma)

          return ep_record, cum_rewards

        for episode in range(episodes // parallel_envs):

          batch_records, batch_Q_s = zip(*ray.get([env_run.remote(self.net.cpu()) for _ in range(parallel_envs)]))

          total_rewards += deque([np.sum([exp.reward for exp in episode]) for episode in batch_records])

          if (target_reward is not None) & (len(total_rewards) == 100):
            if np.array(total_rewards).mean() >= target_reward:
              print(f'Solved at target {target_reward}!')
              self.save_model(np.array(total_rewards).mean(), save_from, save_interval, best=True)
              break

          if (target_reward is not None) & (len(total_rewards) == 100):
            self.save_model(np.array(total_rewards).mean(), save_from, save_interval, best=False)
            if np.array(total_rewards).mean() > self.current_best_reward:
              self.current_best_reward = np.array(total_rewards).mean()

          batch_Q_s = torch.tensor([Q_s for episode in batch_Q_s for Q_s in episode],
                                   dtype=torch.float32).to(self.device).unsqueeze(-1)

          zip(*[(exp.state, exp.action) for episode in batch_records for exp in episode])

          batch_states, batch_actions= zip(*[(exp.state, exp.action) for episode in batch_records for exp in episode])

          batch_states = np.array(batch_states)
          batch_actions = torch.stack(batch_actions).to(self.device).unsqueeze(-1)

          self.net.to(self.device)
          _, batch_action_probs, batch_action_logprobs, batch_state_values = self.net(batch_states)

          # calculate loss
          self.optimizer.zero_grad()
          loss = calc_loss_actor_critic(batch_Q_s, batch_actions, batch_action_probs, batch_action_logprobs,
                                        batch_state_values, device=self.device)

          loss.backward()

          # update the model and predict_model
          self.optimizer.step()

          total_loss.append(loss.item())

          print(
            f'Episodes: {episode * parallel_envs}, Loss {np.array(total_loss).mean()}, Mean Reward: {np.array(total_rewards).mean()}')
