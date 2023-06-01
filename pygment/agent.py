import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import time
import datetime as dt
import os
import shutil
from copy import deepcopy
from .actions import GreedyEpsilonSelector, calc_loss_batch, calc_loss_policy, calc_cum_rewards, \
    calc_entropy_loss_policy, calc_loss_actor_critic, calc_iql_loss_batch, calc_iql_policy_loss_batch
from .net import BaseNet, DualNet, ActorCriticNet, PolicyGradientNet, ActorCriticNetContinuous
from .env import wrap_env
from collections import deque
import ray


class Experience:
    '''
  The Experience class stores values from a training episode (values from the environment
  and values from the network).
  '''

    def __init__(self, state=None, action=None, reward=None, cum_reward=None, next_state=None, next_action=None,
                 done=None, action_probs=None, action_logprobs=None, state_value=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.cum_reward = cum_reward
        self.next_state = next_state
        self.next_action = next_action
        self.done = done
        self.action_probs = action_probs
        self.action_logprobs = action_logprobs
        self.state_value = state_value


class BaseAgent:
    """
    Base Agent stores universal attributes and super() methods across the agents.
    """

    def __init__(self, device='cpu'):
        self.net = None
        self.device = device
        self.optimizer = None
        self.env = None
        self.current_best_reward = -10 ** 100
        now = dt.datetime.now()
        self.path = f'./{now.year}_{now.month}_{now.day:02}_{now.hour:02}{now.minute:02}{now.second:02}'
        self._optimizer = None
        self._learning_rate = None
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
            self.device = 'cpu'
            self.optimizer = None
            self.env = None
            self.current_best_reward = -10 ** 100
            now = dt.datetime.now()
            self.path = f'./{now.year}_{now.month}_{now.day:02}_{now.hour:02}_{now.minute:02}_{now.second:02}/'
            self._optimizer = None
            self._learning_rate = None
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

    def train(self, gamma, custom_params=None):
        self.method_check(env_loaded=True, net_exists=True, compiled=True)
        self._gamma = gamma

        if custom_params:
            self.optimizer = self._optimizers[self._optimizer](custom_params)
        else:
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
            # Check directory exists and create if not
            if not os.path.isdir(self.path):
                os.mkdir(self.path)

            # It may be the case that a previous model was loaded and training was continued. If this hasn't been
            # checked yet, screen the previous checkpoints to check whether any were saved as 'model_best_',
            # and clarify the score of that checkpoint.
            if self._screen_files:
                no_best = True
                for file in os.listdir(self.path):
                    if 'model_best_' in file:
                        no_best = False
                        prev_score = file.split('model_best_')[1].split('.pt')[0]

                        # If the score of this checkpoint is below the current model, remove the 'model_best_' from that
                        # checkpoint.
                        if int(float(prev_score)) < (average_reward // save_interval * save_interval):
                            os.rename(self.path + f'/{file}', self.path + f'/model_{int(float(prev_score))}.pt')
                            self._screen_files = False
                            break

                if no_best:
                    self._screen_files = False

            # Define the directory to save the model in, and the name of the file, which follows the template of
            # 'model_(score)' initially, and 'model_best_(score)' at the terminus.
            current_model_path = self.path + f'/model_best_{int(average_reward // save_interval * save_interval)}.pt' if best \
                else self.path + f'/model_{int(average_reward // save_interval * save_interval)}.pt'

            # For some reason there are issues with saving and re-training the model under mps, so (for the time being)
            # we create a deepcopy of our model to get around this.
            current_model = deepcopy(self.net)
            torch.save(current_model, current_model_path)

    def load_model(self, path=None):
        if path is None:
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

        self.path = os.path.dirname(path)


class DQNAgent(BaseAgent):
    """
    DQNAgent is a Double DQN agent which learns from its environment using a deep Q network (with both a
    current network and an intermittently synchronised target network).
    """

    def __init__(self, device):
        super().__init__(device)
        self._buffer_size = None
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

        self.net.add_layers(nodes, self.env)
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

    def train(self, target_reward, episodes=10000, batch_size=64, gamma=0.99, buffer=10000,
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

        for episode in range(1, episodes + 1):
            if episode % 5 == 0:
                print(
                    f'Episodes completed: {episode}, Loss: {np.array(total_loss).mean()}, Reward: {np.array(total_rewards).mean()}')

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


class IQLAgent(BaseAgent):
    """
    IQLAgent is an implementation of the Implicit Q-Learning algorithm, a completely off-line RL algorithm.
    """

    def __init__(self, device):
        super().__init__(device)
        self._beta = None
        self._tau = None
        self.net = ActorCriticNet()
        self._batch_size = None

    def reset(self):
        reset_done = super().reset()

        if reset_done:
            self.net = ActorCriticNet()
            self._batch_size = None

    '''
    Feed the desired number of nodes and the associated input/output 
    features to the DualNet network-generating method.
    '''

    def add_network(self, nodes: list):
        super().network_check(nodes)

        self.net.add_layers(nodes, self.env)
        self.net.has_net = True

    def compile(self, optimizer, learning_rate=0.001, weight_decay=1e-5):
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._regularisation = weight_decay
        super().compile_check()

        self.net.to(self.device)

        self._compiled = True

    def train_qv(self, data: list, epochs=1000, batch_size=64, gamma=0.99, tau=0.99, save=False):
        """
        Variable 'data' should be a 1D list of Experiences - sorted or unsorted. The reward value should be the
        correct Q_s value for that state i.e., the cumulated discounted reward from that state onwards.
        """

        super().train(gamma)
        self.method_check(env_loaded=True, net_exists=True, compiled=True)
        self._gamma = gamma
        self._tau = tau
        self._batch_size = batch_size

        if save:
            if not os.path.isdir(self.path):
                os.makedirs(self.path)

        # Training the Q/V network
        print('Beginning training of Q/V networks...\n')
        old_qv_loss = 10 ** 10
        counter = 0
        backup_net = deepcopy(self.net)
        for epoch in range(1, epochs + 1):
            np.random.default_rng().shuffle(data)
            current_loss = []
            current_loss_q = []
            current_loss_v = []

            for i in tqdm(range(0, len(data), batch_size)):
                batch = np.array(data[i:i + batch_size])

                self.optimizer.zero_grad()

                loss_q, loss_v = calc_iql_loss_batch(batch, self.device, self.net, self._gamma, self._tau)

                loss = loss_q + loss_v
                loss.backward()
                self.optimizer.step()

                current_loss.append(loss.item())
                current_loss_q.append(loss_q.item())
                current_loss_v.append(loss_v.item())

            if np.array(current_loss).mean() > old_qv_loss:
                counter += 1
                if counter == 5:
                    self.net = deepcopy(backup_net)
                    self.optimizer = self._optimizers[self._optimizer](self.net.parameters(), lr=self._learning_rate,
                                                                       weight_decay=self._regularisation)
                    print(f'Epoch {epoch}: Loss has increased from {old_qv_loss} to {np.array(current_loss).mean()} five times in a row. '
                          f'Loss_Q: {np.array(current_loss_q).mean()}, Loss_V: {np.array(current_loss_v).mean()}\n'
                          f'Reverting to old net and stopping Q/V training.')
                    break
                else:
                    print(
                        f'Epoch {epoch}: Loss has increased from {old_qv_loss} to {np.array(current_loss).mean()}. '
                        f'Loss_Q: {np.array(current_loss_q).mean()}, Loss_V: {np.array(current_loss_v).mean()}')
            else:
                counter = 0
                backup_net = deepcopy(self.net)
                if save:
                    old_save_path = os.path.join(self.path, f'no_policy_model_loss_{round(old_qv_loss, 3)}.pt')
                    new_save_path = os.path.join(self.path,
                                                 f'no_policy_model_loss_{round(np.array(current_loss).mean(), 3)}.pt')

                    torch.save(self.net, new_save_path)
                    if os.path.isfile(old_save_path) and old_save_path != new_save_path:
                        os.remove(old_save_path)

                old_qv_loss = np.array(current_loss).mean()

                print(
                    f'Epochs completed: {epoch}, Loss: {np.array(current_loss).mean()}, '
                    f'Loss_Q: {np.array(current_loss_q).mean()}, Loss_V: {np.array(current_loss_v).mean()}')

        print('Q/V training complete!')

    def train_policy(self, data: list, epochs=1000, batch_size=64, beta=1, save=False):
        """
        Variable 'data' should be a 1D list of Experiences - sorted or unsorted. The reward value should be the
        correct Q_s value for that state i.e., the cumulated discounted reward from that state onwards.
        """
        super().train(self._gamma)

        self.method_check(env_loaded=True, net_exists=True, compiled=True)
        self._beta = beta
        self._batch_size = batch_size

        if save:
            if not os.path.isdir(self.path):
                os.makedirs(self.path)

        # Training the policy network
        print('Beginning training of policy network...\n')
        old_policy_loss = 10 ** 10
        counter = 0
        backup_net = deepcopy(self.net)
        for epoch in range(1, epochs + 1):
            np.random.default_rng().shuffle(data)
            current_loss = []

            for i in tqdm(range(0, len(data), batch_size)):
                batch = np.array(data[i:i + batch_size])

                self.optimizer.zero_grad()

                loss = calc_iql_policy_loss_batch(batch, self.device, self.net, self._beta)

                loss.backward()
                self.optimizer.step()

                current_loss.append(loss.item())

            if np.array(current_loss).mean() > old_policy_loss:
                counter += 1
                if counter == 5:
                    self.net = deepcopy(backup_net)
                    self.optimizer = self._optimizers[self._optimizer](self.net.parameters(), lr=self._learning_rate,
                                                                       weight_decay=self._regularisation)
                    print(f'Epoch {epoch}: Loss has increased from {old_policy_loss} to {np.array(current_loss).mean()} five times in a row. \n'
                          f'Reverting to old net and ending policy training.')
                    break
                else:
                    print(
                        f'Epoch {epoch}: Loss has increased from {old_policy_loss} to {np.array(current_loss).mean()}.')
            else:
                counter = 0
                backup_net = deepcopy(self.net)
                if save:
                    old_save_path = os.path.join(self.path, f'policy_model_loss_{round(old_policy_loss, 3)}.pt')
                    new_save_path = os.path.join(self.path, f'policy_model_loss_{round(np.array(current_loss).mean(), 3)}.pt')

                    torch.save(self.net, new_save_path)
                    if os.path.isfile(old_save_path) and old_save_path != new_save_path:
                        os.remove(old_save_path)

                old_policy_loss = np.array(current_loss).mean()

                print(
                    f'Epochs completed: {epoch}, Loss: {np.array(current_loss).mean()}')

        print('Policy training complete!')

    def evaluate(self, episodes=100, parallel_envs=32):

        @ray.remote
        def env_run(predict_net):
            done = False
            prem_done = False
            state = self.env.reset()[0]
            ep_record = []
            while not done and not prem_done:
                with torch.no_grad():
                    Q_s, _ = predict_net.forward(state, device='cpu')

                action = torch.argmax(Q_s)

                next_state, reward, done, prem_done, _ = self.env.step(action.item())

                ep_record.append(Experience(state=state,
                                            action=action,
                                            reward=reward,
                                            next_state=next_state))

                state = next_state

            total_reward = sum([exp.reward for exp in ep_record])

            return ep_record, total_reward

        all_states = []
        all_actions = []
        all_rewards = []
        all_total_rewards = []
        all_next_states = []

        print(f'Beginning evaluation over {episodes} episodes...\n')
        for episode in tqdm(range(episodes // parallel_envs)):
            temp_batch_records, temp_total_reward = zip(
                *ray.get([env_run.remote(self.net.cpu()) for _ in range(parallel_envs)]))

            temp_batch_states, temp_batch_actions, temp_batch_rewards, temp_batch_next_states = zip(
                *[(exp.state, exp.action, exp.reward, exp.next_state)
                  for episode in temp_batch_records for exp in episode])

            all_states += list(temp_batch_states)
            all_actions += list(temp_batch_actions)
            all_rewards += list(temp_batch_rewards)
            all_next_states += list(temp_batch_next_states)
            all_total_rewards += list(temp_total_reward)

        print(f'Evaluation complete! Average reward per episode: {np.array(all_total_rewards).mean()}')

        return np.array(all_states), np.array(all_actions), np.array(all_rewards), \
            np.array(all_next_states), np.array(all_total_rewards)

    def choose_action(self, state, device='cpu'):
        with torch.no_grad():
            Q_s, _ = self.net.forward(state, device=device)

        action = torch.argmax(Q_s)

        return action.item()


class PolicyGradient(BaseAgent):
    """
    PolicyGradient is a neural network based on REINFORCE
    """

    def __init__(self, device):
        super().__init__(device)
        self.net = PolicyGradientNet()

    def reset(self):
        reset_done = super().reset()

        if reset_done:
            self.net = PolicyGradientNet()

    def add_network(self, nodes: list):
        super().network_check(nodes)

        self.net.add_layers(nodes, self.env)
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

    def train(self, target_reward=None, episodes=10000, ep_update=4, gamma=0.99):
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

            state_records, action_records = zip(
                *[(exp.state, exp.action) for buffer in buffer_record for exp in buffer])
            cum_rewards = [cum_r for record in cum_reward for cum_r in record]
            self.net.to('cpu')
            _, action_probs_records, action_logprobs_records = self.net(state_records, self.device)

            # calculate loss
            self.optimizer.zero_grad()
            loss = calc_loss_policy(cum_rewards, action_records, action_logprobs_records, device='cpu')
            loss.backward(retain_graph=True)
            # Now calculate the entropy loss

            entropy_loss = calc_entropy_loss_policy(action_probs_records, action_logprobs_records, device='cpu')
            entropy_loss.backward()
            # update the model and predict_model
            self.optimizer.step()
            total_loss.append(loss.item())

            print(
                f'Episodes: {episode * ep_update}, Loss {np.array(total_loss).mean()}, Mean Reward: {np.array(total_rewards).mean()}')

    def action_selector(self, obs):
        self.method_check(env_loaded=True, net_exists=True, compiled=True)

        return self.net(obs)[0]


class PPO(BaseAgent):
    """
    ActorCritic uses the actor-critic neural network to solve environments with a discrete action space.
    """

    def __init__(self, device):
        super().__init__(device)
        self.net = ActorCriticNet()

    def reset(self):
        reset_done = super().reset()

        if reset_done:
            self.net = ActorCriticNet()

    def add_network(self, nodes: list):
        super().network_check(nodes)

        self.net.add_layers(nodes, self.env)
        self.net.has_net = True

    def compile(self, optimizer, learning_rate=0.001, weight_decay=1e-5, clip=1, lower_clip=None, upper_clip=None):
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._regularisation = weight_decay
        super().compile_check()

        # self.net.to(self.device)

        '''for p in self.net.parameters():
            p.register_hook(lambda grad: torch.clamp(grad,
                                                     lower_clip if lower_clip is not None else -clip,
                                                     upper_clip if upper_clip is not None else clip))'''

        self._compiled = True

    def choose_action(self, state, device='cpu'):
        action_logits, state_value = self.net.forward(state, device=device)

        action, _, _ = self.get_action_and_logprobs(action_logits)

        return action.item()

    def get_action_and_logprobs(self, action_logits, action=None):
        action_probs = F.softmax(action_logits, dim=-1)
        action_logprobs = F.log_softmax(action_logits, dim=-1)
        action_distribution = Categorical(action_probs)

        if len(action_probs.shape) == 1:
            entropy = (action_probs * action_logprobs).sum()
        else:
            entropy = (action_probs * action_logprobs).sum(1).mean()

        # Following is to avoid rare events where probability is represented as zero (and logprob = inf),
        # but is in fact non-zero, and an action is sampled from this index.
        if action is None:
            while True:
                action = action_distribution.sample()
                if action.shape:
                    if ~torch.isinf(action_logprobs.gather(1, action.unsqueeze(-1)).squeeze(-1)).all():
                        break
                else:
                    if ~torch.isinf(action_logprobs[action.item()]):
                        break

        if action.shape:
            action_logprobs = action_logprobs.gather(1, action.reshape(-1, 1)).squeeze(-1)
        else:
            action_logprobs = action_logprobs[action.item()]

        return action, action_logprobs, entropy

    def train(self, target_reward, save_from, save_interval=10, episodes=10000, parallel_envs=32, update_iter=4,
              update_steps=1000, batch_size=128, gamma=0.99):
        super().train(gamma)

        total_rewards = deque([], maxlen=parallel_envs)
        total_loss = deque([], maxlen=update_steps)
        total_policy_loss = deque([], maxlen=update_steps)
        total_value_loss = deque([], maxlen=update_steps)
        list_of_indexes = np.array([i for i in range(update_steps)])

        @ray.remote
        def env_run(predict_net):
            done = False
            prem_done = False
            state = self.env.reset()[0]
            ep_record = []
            while not done and not prem_done:
                with torch.no_grad():
                    action_logits, state_value = predict_net.forward(state, device='cpu')

                action, old_policy_logprobs, _ = self.get_action_and_logprobs(action_logits)

                next_state, reward, done, prem_done, _ = self.env.step(action.item())

                ep_record.append(Experience(state=state,
                                            action=action,
                                            reward=reward,
                                            action_logprobs=old_policy_logprobs))

                state = next_state

            cum_rewards = calc_cum_rewards([exp.reward for exp in ep_record], self._gamma)

            return ep_record, cum_rewards

        for episode in range(episodes):
            batch_records = []
            batch_Q_s = []
            batch_states = []
            batch_actions = []
            batch_old_policy_logprobs = []
            while True:
                temp_batch_records, temp_batch_Q_s = zip(
                    *ray.get([env_run.remote(self.net.cpu()) for _ in range(parallel_envs)]))

                total_rewards += deque([np.sum([exp.reward for exp in episode]) for episode in temp_batch_records])

                batch_Q_s += [Q_s for episode in temp_batch_Q_s for Q_s in episode]

                temp_batch_states, temp_batch_actions, temp_batch_old_policy_logprobs = zip(*[(exp.state,
                                                                                               exp.action,
                                                                                               exp.action_logprobs)
                                                                                              for episode in
                                                                                              temp_batch_records for exp
                                                                                              in
                                                                                              episode])
                batch_states += list(temp_batch_states)
                batch_actions += list(temp_batch_actions)
                batch_old_policy_logprobs += list(temp_batch_old_policy_logprobs)

                if len(batch_states) >= update_steps:
                    break

            if (target_reward is not None) & (len(total_rewards) == parallel_envs):
                if np.array(total_rewards).mean() >= target_reward:
                    print(f'Solved at target {target_reward}!')
                    self.save_model(np.array(total_rewards).mean(), save_from, save_interval, best=True)
                    break

            if (target_reward is not None) & (len(total_rewards) == parallel_envs):
                self.save_model(np.array(total_rewards).mean(), save_from, save_interval, best=False)
                if np.array(total_rewards).mean() > self.current_best_reward:
                    self.current_best_reward = np.array(total_rewards).mean()

            list_of_all_steps = np.array([i for i in range(len(batch_states))])
            choice_of_all_steps = np.random.choice(list_of_all_steps,
                                                   size=update_steps)

            batch_Q_s = torch.tensor(batch_Q_s, dtype=torch.float32).to(self.device).unsqueeze(-1)[choice_of_all_steps]

            batch_states = np.array(batch_states)[choice_of_all_steps]
            batch_actions = torch.stack(batch_actions).to(self.device).unsqueeze(-1)[choice_of_all_steps]
            batch_old_policy_logprobs = torch.stack(batch_old_policy_logprobs).to(self.device)[choice_of_all_steps]

            self.net.to(self.device)

            for _ in range(update_iter):

                with torch.no_grad():
                    _, batch_state_values = self.net(batch_states, self.device)

                advantage = batch_Q_s - batch_state_values
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                list_sample_idxes = np.random.choice(list_of_indexes,
                                                     size=(update_steps // batch_size, batch_size),
                                                     replace=False).tolist()
                # The following finds the leftover idxes - this is ordered, but because it forms a minibatch,
                # this shouldn't matter / doesn't need shuffling.
                remaining_idxes = np.setdiff1d(list_of_indexes, list_sample_idxes).tolist()
                list_sample_idxes.append(remaining_idxes)

                for sample_idxes in list_sample_idxes:
                    batch_action_logits, batch_state_values = self.net(batch_states[sample_idxes],
                                                                       self.device)

                    _, batch_action_logprobs, batch_entropy = self.get_action_and_logprobs(batch_action_logits,
                                                                                           batch_actions[sample_idxes])

                    # calculate loss
                    self.optimizer.zero_grad()
                    policy_loss, value_loss, entropy_loss = calc_loss_actor_critic(batch_Q_s[sample_idxes],
                                                                                   batch_actions[sample_idxes],
                                                                                   batch_entropy, batch_action_logprobs,
                                                                                   batch_state_values,
                                                                                   device=self.device,
                                                                                   batch_old_policy_logprobs=
                                                                                   batch_old_policy_logprobs[
                                                                                       sample_idxes],
                                                                                   advantage=advantage[sample_idxes])

                    loss = policy_loss + value_loss - entropy_loss
                    loss.backward()

                    total_loss.append(policy_loss.item() + value_loss.item())
                    total_policy_loss.append(policy_loss.item())
                    total_value_loss.append(value_loss.item())

                    # update the model and predict_model
                    self.optimizer.step()

            print(
                f'Epoch: {episode}, '
                f'Loss {round(np.array(total_loss).mean(), 2)} '
                f'(Policy {round(np.array(total_policy_loss).mean(), 2)}, '
                f'Value {round(np.array(total_value_loss).mean(), 2)}), '
                f'Mean Reward: {round(np.array(total_rewards).mean(), 2)}')

    def explore(self, episodes=10000, parallel_envs=32, gamma=0.99):
        super().train(gamma)

        @ray.remote
        def env_run(predict_net):
            done = False
            prem_done = False
            state = self.env.reset()[0]
            ep_record = []
            while not done and not prem_done:
                with torch.no_grad():
                    action_logits, state_value = predict_net.forward(state, device='cpu')

                action, old_policy_logprobs, _ = self.get_action_and_logprobs(action_logits)

                next_state, reward, done, prem_done, _ = self.env.step(action.item())

                ep_record.append(Experience(state=state,
                                            action=action,
                                            reward=reward,
                                            next_state=next_state))

                state = next_state

            cum_rewards = calc_cum_rewards([exp.reward for exp in ep_record], self._gamma)

            return ep_record, cum_rewards

        all_states = []
        all_actions = []
        all_rewards = []
        all_cum_rewards = []
        all_next_states = []

        print(f'Beginning exploration over {episodes} episodes...')
        for episode in tqdm(range(episodes // parallel_envs)):
            temp_batch_records, temp_batch_Qs = zip(
                *ray.get([env_run.remote(self.net.cpu()) for _ in range(parallel_envs)]))

            temp_batch_states, temp_batch_actions, temp_batch_rewards, temp_batch_next_states = zip(
                *[(exp.state, exp.action, exp.reward, exp.next_state)
                  for episode in temp_batch_records for exp in episode])

            all_states += list(temp_batch_states)
            all_actions += list(temp_batch_actions)
            all_rewards += list(temp_batch_rewards)
            all_next_states += list(temp_batch_next_states)
            all_cum_rewards += list([Q_s for episode in temp_batch_Qs for Q_s in episode])

        return np.array(all_states), np.array(all_actions), np.array(all_rewards), \
            np.array(all_next_states), np.array(all_cum_rewards)


class PPOContinuous(BaseAgent):
    """
  PPOContinuous uses the actor-critic neural network to solve environments with a continuous action space.
  """

    def __init__(self, device):
        super().__init__(device)
        self.net = ActorCriticNetContinuous()

    def reset(self):
        reset_done = super().reset()

        if reset_done:
            self.net = ActorCriticNetContinuous()

    def add_network(self, nodes: list):
        super().network_check(nodes)

        self.net.add_layers(nodes, self.env)
        self.net.has_net = True

    def compile(self, optimizer, learning_rate=0.001, weight_decay=1e-5, clip=1, lower_clip=None, upper_clip=None):
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._regularisation = weight_decay
        super().compile_check()

        # self.net.to(self.device)

        for p in self.net.parameters():
            p.register_hook(lambda grad: torch.clamp(grad,
                                                     lower_clip if lower_clip is not None else -clip,
                                                     upper_clip if upper_clip is not None else clip))

        self._compiled = True

    def get_action_and_logprobs(self, action_means, action_stds, action=None):
        dist = Normal(action_means, action_stds)
        if action is None:
            action = dist.sample()

        action_logprobs = dist.log_prob(action)
        entropy = dist.entropy().mean()

        if action.shape[0] == 1:
            action = action.reshape(-1)
            action_logprobs = action_logprobs.reshape(-1)

        return action, entropy, action_logprobs

    def train(self, target_reward, save_from, save_interval=10, episodes=10000, parallel_envs=32, update_iter=4,
              update_steps=1000, batch_size=128, gamma=0.99, epsilon=0.2):
        custom_params = [{'params': self.net.actor_net.parameters(),
                          'lr': 0.0005},  # 0.0005
                         {'params': self.net.critic_net.parameters(),
                          'lr': 0.0005}]  # 0.0005
        super().train(gamma, custom_params)

        total_rewards = deque([], maxlen=parallel_envs)
        total_loss = deque([], maxlen=update_steps)
        total_policy_loss = deque([], maxlen=update_steps)
        total_value_loss = deque([], maxlen=update_steps)
        list_of_indexes = np.array([i for i in range(update_steps)])

        @ray.remote
        def env_run(predict_net):
            done = False
            prem_done = False
            state = self.env.reset()[0]
            ep_record = []
            episode_reward = 0
            while not done and not prem_done:
                with torch.no_grad():
                    action_means, action_stds, state_value = predict_net.forward(state, device='cpu')

                action, _, old_policy_logprobs = self.get_action_and_logprobs(action_means,
                                                                              action_stds)

                next_state, reward, done, prem_done, _ = self.env.step(torch.tanh(action).numpy())

                ep_record.append(Experience(state=state,
                                            action=action,
                                            reward=reward,
                                            action_logprobs=old_policy_logprobs))

                episode_reward += reward

                state = next_state

            episode_states, episode_actions, episode_old_policy_logprobs = zip(*[(exp.state,
                                                                                  exp.action,
                                                                                  exp.action_logprobs)
                                                                                 for exp in ep_record])

            episode_states = np.array(episode_states)
            episode_actions = torch.stack(episode_actions)
            episode_old_policy_logprobs = torch.stack(episode_old_policy_logprobs)

            cum_rewards = calc_cum_rewards([exp.reward for exp in ep_record], self._gamma)

            return episode_states, episode_actions, episode_old_policy_logprobs, cum_rewards, episode_reward

        for episode in range(episodes):
            batch_Q_s = []
            step = 0
            while True:
                step += 1
                predict_net = ray.put(self.net.cpu())
                temp_batch_states, temp_batch_actions, temp_batch_old_policy_logprobs, temp_batch_Q_s, batch_rewards = zip(
                    *ray.get([env_run.remote(predict_net) for _ in range(parallel_envs)]))
                del predict_net

                total_rewards += deque(list(batch_rewards))

                batch_Q_s += [Q_s for episode in temp_batch_Q_s for Q_s in episode]

                temp_batch_states = np.concatenate(tuple([i for i in temp_batch_states]))
                temp_batch_actions = torch.concatenate(tuple([i for i in temp_batch_actions]))
                temp_batch_old_policy_logprobs = torch.concatenate(tuple([i for i in temp_batch_old_policy_logprobs]))

                if step == 1:
                    batch_states = np.array(temp_batch_states)
                    batch_actions = temp_batch_actions
                    batch_old_policy_logprobs = temp_batch_old_policy_logprobs
                else:
                    batch_states = np.concatenate((batch_states, temp_batch_states))
                    batch_actions = torch.concatenate((batch_actions, temp_batch_actions))
                    batch_old_policy_logprobs = torch.concatenate((batch_old_policy_logprobs,
                                                                   temp_batch_old_policy_logprobs))

                if len(batch_states) >= update_steps:
                    break

            parallel_envs *= step

            if (target_reward is not None) & (len(total_rewards) == total_rewards.maxlen):
                if np.array(total_rewards).mean() >= target_reward:
                    print(f'Solved at target {target_reward}!')
                    self.save_model(np.array(total_rewards).mean(), save_from, save_interval, best=True)
                    break

            if (target_reward is not None) & (len(total_rewards) == total_rewards.maxlen):
                self.save_model(np.array(total_rewards).mean(), save_from, save_interval, best=False)
                if np.array(total_rewards).mean() > self.current_best_reward:
                    self.current_best_reward = np.array(total_rewards).mean()

            list_of_all_steps = np.array([i for i in range(len(batch_states))])
            choice_of_all_steps = np.random.choice(list_of_all_steps,
                                                   size=update_steps)

            batch_Q_s = torch.tensor(batch_Q_s,
                                     dtype=torch.float32).to(self.device).unsqueeze(-1)[choice_of_all_steps]

            batch_states = batch_states[choice_of_all_steps]
            batch_actions = batch_actions.to(self.device)[choice_of_all_steps]
            batch_old_policy_logprobs = batch_old_policy_logprobs.to(self.device)[choice_of_all_steps]

            self.net.to(self.device)

            for _ in range(update_iter):

                with torch.no_grad():
                    _, _, batch_state_values = self.net(batch_states, self.device)

                advantage = batch_Q_s - batch_state_values
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                list_sample_idxes = np.random.choice(list_of_indexes,
                                                     size=(update_steps // batch_size, batch_size),
                                                     replace=False).tolist()
                # The following finds the leftover idxes - this is ordered, but because it forms a minibatch,
                # this shouldn't matter / doesn't need shuffling.
                remaining_idxes = np.setdiff1d(list_of_indexes, list_sample_idxes).tolist()
                list_sample_idxes.append(remaining_idxes)

                for sample_idxes in list_sample_idxes:
                    if sample_idxes == []:
                        continue
                    batch_action_means, batch_action_stds, batch_state_values = self.net(batch_states[sample_idxes],
                                                                                         self.device)

                    _, batch_entropy, batch_action_logprobs = self.get_action_and_logprobs(batch_action_means,
                                                                                           batch_action_stds,
                                                                                           batch_actions[sample_idxes])

                    # calculate loss
                    self.optimizer.zero_grad()
                    policy_loss, value_loss, entropy_loss = calc_loss_actor_critic(batch_Q_s[sample_idxes],
                                                                                   batch_actions[sample_idxes],
                                                                                   batch_entropy, batch_action_logprobs,
                                                                                   batch_state_values,
                                                                                   device=self.device,
                                                                                   batch_old_policy_logprobs=
                                                                                   batch_old_policy_logprobs[
                                                                                       sample_idxes],
                                                                                   advantage=advantage[sample_idxes],
                                                                                   epsilon=epsilon)

                    loss = policy_loss + 0.5 * value_loss  # - entropy_loss
                    loss.backward()

                    total_loss.append(
                        (-batch_action_logprobs * advantage[sample_idxes]).mean().item() + value_loss.item())
                    total_policy_loss.append((-batch_action_logprobs * advantage[sample_idxes]).mean().item())
                    total_value_loss.append(value_loss.item())

                    # update the model and predict_model
                    self.optimizer.step()

            print(
                f'Epoch: {episode}, '
                f'Loss {round(np.array(total_loss).mean(), 2)} '
                f'(Policy {round(np.array(total_policy_loss).mean(), 2)}, '
                f'Value {round(np.array(total_value_loss).mean(), 2)}), '
                f'Mean Reward: {round(np.array(total_rewards).mean(), 2)}')


'''
NEXT STEP - START CONSOLIDATING INCONSISTENCIES ACROSS CLASSES IN TERMS OF NET AND GET_ACTION
NOTE THAT ANIMATE CURRENTLY IS NOT CONSISTENT ACROSS ALL CLASSES
THEN - CONSIDER CREATING INDIVIDUAL ACTOR AND CRITIC NET CLASSES
'''
