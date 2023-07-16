import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import time
import sys
import datetime as dt
import os
import shutil
from copy import deepcopy
from .actions import GreedyEpsilonSelector, calc_loss_batch, calc_loss_policy, calc_cum_rewards, \
    calc_entropy_loss_policy, calc_loss_actor_critic, calc_iql_q_loss_batch, calc_iql_v_loss_batch, \
    calc_iql_policy_loss_batch
from .net import BaseNet, DualNet, ActorCriticNet, PolicyGradientNet, ActorCriticNetContinuous, CriticNet, ActorNet
from .env import wrap_env
from collections import deque
import ray
from math import ceil
import matplotlib.pyplot as plt


class Experience:
    '''
  The Experience class stores values from a training episode (values from the environment
  and values from the network).
  '''

    def __init__(self, state=None, action=None, reward=None, cum_reward=None, next_state=None, next_action=None,
                 done=None, action_probs=None, action_logprobs=None, state_value=None, original_reward=None,
                 original_cumreward=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.original_reward = original_reward
        self.original_cumreward = original_cumreward
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

    def __init__(self, device='cpu', path=None):
        self.net = None
        self.device = device
        self.optimizer = None
        self.env = None
        self.current_best_reward = -10 ** 100
        now = dt.datetime.now()
        if path is None:
            self.path = f'./{now.year}_{now.month}_{now.day:02}_{now.hour:02}{now.minute:02}{now.second:02}'
        else:
            self.path = os.path.join(path,
                                     f'{now.year}_{now.month}_{now.day:02}_{now.hour:02}{now.minute:02}{now.second:02}')
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

    def train_base(self, gamma, custom_params=None):
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

    def __init__(self, device, path=None):
        super().__init__(device, path)
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

    def __init__(self, device, path=None):
        super().__init__(device, path)
        self._alpha = None
        self.custom_params = None
        self._beta = None
        self._tau = None
        self.value = CriticNet()
        self.critic1 = DualNet()
        self.critic2 = DualNet()
        self.actor = ActorNet()
        self.behaviour_policy = ActorNet()
        self.net = self.critic1
        self._batch_size = None

    def reset(self):
        reset_done = super().reset()

        if reset_done:
            self._alpha = None
            self.custom_params = None
            self._beta = None
            self._tau = None
            self.value = CriticNet()
            self.critic1 = DualNet()
            self.critic2 = DualNet()
            self.actor = ActorNet()
            self.behaviour_policy = ActorNet()
            self.net = self.actor
            self._batch_size = None

    def add_network(self, nodes: list):
        super().network_check(nodes)

        self.value.add_layers(nodes, self.env)
        self.critic1.add_layers(nodes, self.env)
        self.critic2.add_layers(nodes, self.env)
        self.actor.add_layers(nodes, self.env)
        self.behaviour_policy.add_layers(nodes, self.env)
        self.value.has_net = True
        self.critic1.has_net = True
        self.critic2.has_net = True
        self.actor.has_net = True
        self.behaviour_policy.has_net = True

    def compile(self, optimizer, learning_rate=0.001, weight_decay=1e-5, clip=1.0, lower_clip=None, upper_clip=None):
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._regularisation = weight_decay
        super().compile_check()

        self.value.to(self.device)
        # self.critic1.main_net.to(self.device)
        self.critic1.target_net.to(self.device)
        # self.critic2.main_net.to(self.device)
        self.critic2.target_net.to(self.device)
        self.actor.to(self.device)
        self.behaviour_policy.to(self.device)
        """
        for params in [self.value.parameters(), self.critic1.main_net.parameters(), self.critic2.main_net.parameters(),
                       self.critic1.target_net.parameters(), self.critic2.target_net.parameters(),
                       self.actor.parameters(), self.behaviour_policy.parameters()]:
        """
        for params in [self.value.parameters(), self.critic1.target_net.parameters(),
                       self.critic2.target_net.parameters(), self.actor.parameters(),
                       self.behaviour_policy.parameters()]:
            for p in params:
                p.register_hook(lambda grad: torch.clamp(grad,
                                                         lower_clip if lower_clip is not None else -clip,
                                                         upper_clip if upper_clip is not None else clip))

        self._compiled = True

    @staticmethod
    def sample(data, batch_size):
        idxs = np.random.default_rng().choice(len(data), size=batch_size, replace=False)
        return [data[idx] for idx in idxs]

    def clone_behaviour(self, data, batch_size=64, epochs=10000, evaluate=False, save=False):

        # Set up optimiser
        self.custom_params = []
        self.custom_params.append({'params': self.behaviour_policy.parameters(),
                                   'lr': self._learning_rate,
                                   'weight_decay': self._regularisation})

        super().train_base(0.99, custom_params=self.custom_params)

        # Make save directory if needed
        if save:
            if not os.path.isdir(self.path):
                os.makedirs(self.path)

        # Create logs
        old_policy_loss = torch.inf
        best_policy_loss = torch.inf
        current_loss_policy = []

        # Start training
        print('Beginning training...\n')
        progress_bar = tqdm(range(1, int(epochs) + 1), file=sys.stdout)
        for epoch in progress_bar:
            batch = self.sample(data, batch_size)

            loss = self._update_behaviour_policy(batch)

            current_loss_policy.append(loss)

            if epoch % 1 == 0:
                if evaluate:
                    _, _, _, _, total_rewards = self.evaluate(episodes=1000, parallel_envs=512,
                                                              verbose=False, behaviour=True)

                    with open(
                            '/Users/thomasfrost/Documents/Github/pygment/Informal experiments/clone_behaviour/all_rewards.txt',
                            'a') as f:
                        f.write(f'{int(np.array(total_rewards.mean()))} ')
                        f.close()

                    with open(
                            '/Users/thomasfrost/Documents/Github/pygment/Informal experiments/clone_behaviour/policy_loss.txt',
                            'a') as f:
                        f.write(f'{round(np.array(current_loss_policy).mean(), 6)} ')
                        f.close()

                print(f'\nSteps completed: {epoch}\n')
                print(
                    f'Behaviour policy loss {"decreased" if np.array(current_loss_policy).mean() < old_policy_loss else "increased"} '
                    f'from {old_policy_loss} to {round(np.array(current_loss_policy).mean(), 5)}'
                )
                print(f'Best policy loss: {min(best_policy_loss, round(np.array(current_loss_policy).mean(), 5))}')
                if evaluate:
                    print(f'Average reward: {int(np.array(total_rewards).mean())}')

                if np.array(current_loss_policy).mean() < best_policy_loss:
                    if save:
                        for net, name in [
                            [self.behaviour_policy, 'behaviour_policy']
                        ]:
                            old_save_path = os.path.join(self.path, f'{name}_{best_policy_loss}.pt')
                            new_save_path = os.path.join(self.path,
                                                         f'{name}_{round(np.array(current_loss_policy).mean(), 5)}.pt')

                            torch.save(net, new_save_path)
                            if os.path.isfile(old_save_path) and old_save_path != new_save_path:
                                os.remove(old_save_path)

                    best_policy_loss = round(np.array(current_loss_policy).mean(), 5)

                old_policy_loss = round(np.array(current_loss_policy).mean(), 5)
                current_loss_policy = []

    def write_to_txt(self, sample_txt, filename):
        if not os.path.isdir(os.path.join(self.path, os.path.dirname(filename))):
            os.makedirs(os.path.join(self.path, os.path.dirname(filename)))

        if os.path.isfile(os.path.join(self.path, filename + '.txt')):
            with open(os.path.join(self.path, filename + '.txt'), 'a') as f:
                f.write(sample_txt)
                f.close()
        else:
            with open(os.path.join(self.path, filename + '.txt'), 'w') as f:
                f.write(sample_txt)
                f.close()

    def train(self, data, evaluate=True, steps=1000, batch_size=64, stop_early_counter=30,
              gamma=0.99, tau=0.5, alpha=0.01, beta=.1, update_iter=4, ppo_clip=0.01, ppo_clip_decay=0.9, save=False):
        """
        Variable 'data' should be a 1D list of Experiences - sorted or unsorted. The reward value should be the
        correct Q_s value for that state i.e., the cumulated discounted reward from that state onwards.
        """
        # if save:
        # evaluate = True

        # Set up optimiser
        self.method_check(env_loaded=True, net_exists=True, compiled=True)
        self._gamma = gamma

        self.optimizer = {}
        scheduler = {}

        for network_name, network in [
            ['value', self.value],
            ['critic1_target', self.critic1.target_net],
            ['critic2_target', self.critic2.target_net],
            ['actor', self.actor]
        ]:
            self.optimizer[network_name] = self._optimizers[self._optimizer](network.parameters(),
                                                                             lr=self._learning_rate,
                                                                             weight_decay=self._regularisation)

            scheduler[network_name] = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer[network_name],
                                                                                 T_max=steps)

        # Create stochastic weight averaged models
        swa_critic1_target_net = torch.optim.swa_utils.AveragedModel(self.critic1.target_net)
        swa_critic2_target_net = torch.optim.swa_utils.AveragedModel(self.critic2.target_net)
        swa_value = torch.optim.swa_utils.AveragedModel(self.value)
        swa_actor = torch.optim.swa_utils.AveragedModel(self.actor)

        swa_scheduler = {}
        for optim_name in ['critic1_target', 'critic2_target', 'value', 'actor']:
            swa_scheduler[optim_name] = torch.optim.swa_utils.SWALR(self.optimizer[optim_name], swa_lr=0.01)

        # Make save directory if needed
        if save:
            if not os.path.isdir(self.path):
                os.makedirs(self.path)

        # Create logs
        val_counter_v = 1
        val_counter_qt = 0
        val_counter_policy = 0
        best_val_loss_v = torch.inf
        best_val_loss_qt = torch.inf
        best_val_loss_policy = torch.inf
        val_loss_qt = torch.inf
        val_loss_v = torch.inf
        val_loss_policy = torch.inf
        swa_steps = 0

        backup_v_network = deepcopy(self.value.state_dict())
        backup_q1_network = deepcopy(self.critic1.target_net.state_dict())
        backup_q2_network = deepcopy(self.critic2.target_net.state_dict())
        backup_policy_network = deepcopy(self.actor.state_dict())

        plot_loss_lim_min = 0
        plot_loss_lim_max = 20
        plot_rewards_lim_min = 100
        plot_rewards_lim_max = 250
        plot_legend_x = 0.55
        plot_legend_y = 0.75

        # If evaluating, start ray instance
        # if evaluate:
        # ray.init()

        all_idxs = (np.array([idx for idx, exp in enumerate(data) if exp.done]) + 1).tolist()
        all_idxs.insert(0, 0)
        all_idxs = [(idx, next_idx) for idx, next_idx in zip(all_idxs[:-1], all_idxs[1:])]
        np.random.default_rng().shuffle(all_idxs)

        train_data_idxs = all_idxs[:7000]
        val_data_idxs = all_idxs[7000:]
        train_data = [data[idx] for start_idx, end_idx in train_data_idxs for idx in range(start_idx, end_idx)]
        val_data = [data[idx] for start_idx, end_idx in val_data_idxs for idx in range(start_idx, end_idx)]

        # Start training
        print('Beginning training...\n')
        progress_bar = tqdm(range(1, int(steps) + 1), file=sys.stdout)
        for step in progress_bar:
            if not any([val_counter_v, val_counter_qt, val_counter_policy]):
                break
            batch = self.sample(train_data, batch_size)
            """
            UPDATE VALUE NETWORK
            """
            if val_counter_v:
                swa_steps += 1
                loss_v = self._update_v(batch, tau)
                val_loss_v = self._validate_v(val_data, tau)

                self.write_to_txt(f'{round(val_loss_v, 6)} ', 'metadata/val_V_loss')

                if swa_steps <= 50:
                    scheduler['value'].step()

                elif swa_steps % 5 == 0:
                    if val_counter_v:
                        swa_value.update_parameters(self.value)

                    swa_scheduler['value'].step()

                if val_loss_v < best_val_loss_v:
                    best_val_loss_v = val_loss_v
                    val_counter_v = 1
                    backup_v_network = self.value.state_dict()
                else:
                    val_counter_v += 1
                    if val_counter_v > stop_early_counter:
                        print('Validation loss for value network has not improved for 30 steps. Stopping training.')
                        val_counter_v = False
                        val_counter_qt = 1
                        swa_steps = 0
                        self.value.load_state_dict(backup_v_network)

            """
            UPDATE CRITIC NETWORKS
            """
            if val_counter_qt:
                swa_steps += 1
                loss_qt = self._update_q(batch, gamma)
                val_loss_qt = self._validate_q(val_data, gamma)

                self.write_to_txt(f'{round(val_loss_qt, 6)} ', 'metadata/val_Q_loss')

                if swa_steps <= 50:
                    scheduler['critic1_target'].step()
                    scheduler['critic2_target'].step()

                elif swa_steps % 5 == 0:
                    if val_counter_qt:
                        swa_critic1_target_net.update_parameters(self.critic1.target_net)
                        swa_critic2_target_net.update_parameters(self.critic2.target_net)

                    swa_scheduler['critic1_target'].step()
                    swa_scheduler['critic2_target'].step()

                if val_loss_qt < best_val_loss_qt:
                    best_val_loss_qt = val_loss_qt
                    val_counter_qt = 1
                    backup_q1_network = self.critic1.target_net.state_dict()
                    backup_q2_network = self.critic2.target_net.state_dict()
                else:
                    val_counter_qt += 1
                    if val_counter_qt > stop_early_counter:
                        print('Validation loss for critic networks has not improved for 30 steps. Stopping training.')
                        val_counter_qt = False
                        val_counter_policy = 1
                        self.critic1.target_net.load_state_dict(backup_q1_network)
                        self.critic2.target_net.load_state_dict(backup_q2_network)
                        swa_steps = 0

            """
            UPDATE ACTOR NETWORK
            """
            if val_counter_policy:
                swa_steps += 1
                loss_policy = self._update_policy(batch, beta, update_iter, ppo_clip)
                val_loss_policy = self._validate_policy(val_data, beta)

                self.write_to_txt(f'{round(val_loss_policy, 6)} ', 'metadata/val_policy_loss')

                if swa_steps <= 50:
                    scheduler['actor'].step()

                elif swa_steps % 5 == 0:
                    if val_counter_policy:
                        swa_actor.update_parameters(self.actor)

                    swa_scheduler['actor'].step()

                if val_loss_policy < best_val_loss_policy:
                    best_val_loss_policy = val_loss_policy
                    val_counter_policy = 1
                    backup_policy_network = self.actor.state_dict()
                else:
                    val_counter_policy += 1
                    if val_counter_policy > stop_early_counter:
                        print(
                            'Validation loss for actor network has not improved for 30 steps. Stopping training.')
                        val_counter_policy = False
                        self.actor.load_state_dict(backup_policy_network)
                        swa_steps = 0

            """
            EVALUATE POLICY ON ENVIRONMENT - OPTIONAL
            """
            if val_counter_policy and evaluate:
                _, _, _, _, total_rewards = self.evaluate(episodes=1000, parallel_envs=512,
                                                          verbose=False)

                self.write_to_txt(f'{int(np.array(total_rewards.mean()))} ', 'metadata/all_rewards')

            if save:
                for net, name, best_loss, current_loss, trigger in [
                    [self.critic1.target_net, 'critic1_target', best_val_loss_qt, val_loss_qt, val_counter_qt],
                    [self.critic2.target_net, 'critic2_target', best_val_loss_qt, val_loss_qt, val_counter_qt],
                    [self.value, 'value', best_val_loss_v, val_loss_v, val_counter_v],
                    [self.actor, 'actor', best_val_loss_policy, val_loss_policy, val_counter_policy]
                ]:
                    if not trigger:
                        continue

                    elif current_loss > best_loss:
                        continue

                    new_save_path = os.path.join(self.path, f'{name}_{round(best_loss, 5)}.pt')
                    old_save_path = new_save_path

                    for file in os.listdir(self.path):
                        if name in file:
                            old_save_path = os.path.join(self.path, file)

                    torch.save(net, new_save_path)
                    if os.path.isfile(old_save_path) and old_save_path != new_save_path:
                        os.remove(old_save_path)

            """
            PLOT LOSS
            """
            
            all_val_counters = [val_counter_v, val_counter_qt, val_counter_policy]
            history_val_v_loss = np.loadtxt(os.path.join(self.path, 'metadata/val_V_loss.txt'), ndmin=1)
            history_val_q_loss = np.array([]) if val_counter_v else np.loadtxt(os.path.join(self.path, 'metadata/val_Q_loss.txt'), ndmin=1)
            history_val_policy_loss = np.array([]) if any(all_val_counters[:2]) else np.loadtxt(os.path.join(self.path, 'metadata/val_policy_loss.txt'), ndmin=1)
            history_policy_rewards = np.array([]) if any(all_val_counters[:2]) else np.loadtxt(os.path.join(self.path, 'metadata/all_rewards.txt'), ndmin=1)
            for plot_number in [1, 2]:
                fig, ax1 = plt.subplots(num=1, clear=True)
                ax2 = ax1.twinx()
                for axes, x_value, y_value, colour, label, linestyle, alpha, which_plots in [
                    [ax1, [(i + 1) for i in range(len(history_val_v_loss))], history_val_v_loss, 'cornflowerblue',
                     'Val V-loss', '-', 1, 1],
                    [ax1, [(i + 1) for i in range(len(history_val_q_loss))], history_val_q_loss, 'violet', 'Val Q Loss',
                     '-', 1, 1],
                    [ax1, [(i + 1) for i in range(len(history_val_policy_loss))], history_val_policy_loss, 'darkseagreen',
                     'Val Policy Loss', '-', 1, 2],
                    [ax2, [(i + 1) for i in range(len(history_policy_rewards))], history_policy_rewards, 'indigo',
                     'Online Reward', 'dotted', 0.4, 2]
                ]:
                    if which_plots >= plot_number:
                        axes.plot(x_value, y_value, color=colour, label=label, linestyle=linestyle, alpha=alpha)

                max_steps_plot = max(len(history_val_v_loss), len(history_val_q_loss), len(history_val_policy_loss)) if plot_number == 1 else len(history_val_policy_loss)
                ax2.hlines(200, xmin=0, xmax=max_steps_plot, color='darkred', linestyle='--', label='Solved',
                           alpha=0.4)
                ax2.hlines(150, xmin=0, xmax=max_steps_plot, color='orange', linestyle='--', label='Behaviour Policy Reward',
                           alpha=0.4)
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc=(plot_legend_x, plot_legend_y))
                plt.title(f'Offline learning, tau = {tau}')
                ax1.set_xlabel('Gradient Steps')
                ax1.set_ylabel('Loss')
                ax2.set_ylabel('Rewards')
                #ax1.set_ylim(plot_loss_lim_min, plot_loss_lim_max)
                ax2.set_ylim(plot_rewards_lim_min, plot_rewards_lim_max)
                if not os.path.isdir(os.path.join(self.path, 'metadata')):
                    os.makedirs(os.path.join(self.path, 'metadata'))
                fig.savefig(os.path.join(self.path, 'metadata/figure.png' if plot_number==1 else 'metadata/figure_rewards_only.png'))

    def _update_q(self, batch: list, gamma):
        """
        Variable 'batch' should be a 1D list of Experiences - sorted or unsorted. The reward value should be the
        correct Q_s value for that state i.e., the cumulated discounted reward from that state onwards.
        """

        # Calculate Q loss
        loss_qt = calc_iql_q_loss_batch(batch, self.device, self.critic1, self.critic2, self.value, gamma)

        # Update Networks
        for network_name in ['critic1_target', 'critic2_target']:
            self.optimizer[network_name].zero_grad()
        loss_qt.backward()
        for network_name in ['critic1_target', 'critic2_target']:
            self.optimizer[network_name].step()

        return loss_qt.item()

    def _update_v(self, batch: list, tau):
        """
        Variable 'batch' should be a 1D list of Experiences - sorted or unsorted. The reward value should be the
        correct Q_s value for that state i.e., the cumulated discounted reward from that state onwards.
        """

        # Calculate V loss
        loss_v = calc_iql_v_loss_batch(batch, self.device, self.value, tau)

        # Update Networks
        self.optimizer['value'].zero_grad()
        loss_v.backward()
        self.optimizer['value'].step()

        return loss_v.item()

    def _update_policy(self, batch: list, beta=1., update_iter=4, ppo_clip=0.01):
        """
        Variable 'batch' should be a 1D list of Experiences - sorted or unsorted. The reward value should be the
        correct Q_s value for that state i.e., the cumulated discounted reward from that state onwards.
        """

        loss_policy = calc_iql_policy_loss_batch(batch, self.device, self.critic1, self.critic2, self.value,
                                                 self.actor, beta)

        if loss_policy:
            self.optimizer['actor'].zero_grad()
            loss_policy.backward()
            self.optimizer['actor'].step()

            return -1 * loss_policy.item()

        else:
            return -1 * torch.inf
        """
        # Unpack the batch
        states, actions = zip(*[(exp.state, exp.action) for exp in batch])

        # Calculate the logprobs of the action taken
        with torch.no_grad():
            old_action_logits = self.actor.forward(states, device=self.device)
            old_action_logprobs = torch.log_softmax(old_action_logits, dim=1)
            old_action_logprobs = old_action_logprobs.gather(1,
                                                             torch.tensor(actions).to(self.device).unsqueeze(
                                                                 -1)).squeeze(
                -1)
            old_action_logprobs = torch.where(torch.isinf(old_action_logprobs), -1000, old_action_logprobs)
            old_action_logprobs = torch.where(old_action_logprobs == 0, -1e-8, old_action_logprobs)

        total_loss_policy = 0

        for _ in range(update_iter):
            loss_policy = calc_iql_policy_loss_batch(batch, self.device, self.critic1, self.critic2,
                                                     self.value, self.actor, old_action_logprobs, beta,
                                                     ppo_clip)

            self.optimizer['actor'].zero_grad()
            loss_policy.backward()
            self.optimizer['actor'].step()

            total_loss_policy += loss_policy.item()

        return total_loss_policy
        """

    def _validate_v(self, val_data: list, tau):
        """
        Variable 'batch' should be a 1D list of Experiences - sorted or unsorted. The reward value should be the
        correct Q_s value for that state i.e., the cumulated discounted reward from that state onwards.
        """

        # Unpack the batch
        states, cum_rewards = zip(*[(exp.state, exp.cum_reward) for exp in val_data])

        with torch.no_grad():
            pred_V_s = self.value.forward(states, device=self.device).squeeze(-1)

        # Calculate loss_v
        loss_v = pred_V_s - torch.tensor(cum_rewards, device=self.device, dtype=torch.float32)
        mask = loss_v > 0
        loss_v = loss_v ** 2
        loss_v[mask] = loss_v[mask] * (1 - tau)
        loss_v[~mask] = loss_v[~mask] * tau
        loss_v = loss_v.mean()

        return loss_v.item()

    def _validate_q(self, val_data: list, gamma):
        # Unpack the batch
        states, actions, reward, next_states, next_actions, cum_rewards, dones = zip(
            *[(exp.state, exp.action, exp.reward,
               exp.next_state, exp.next_action,
               exp.cum_reward, exp.done)
              for exp in val_data])

        with torch.no_grad():
            # Calculate Q_t(s,a) for each state in the batch - this is the 'optimal' Q-function, updated from V(s')
            pred_Q1_t = self.critic1.forward(states, target=True, device=self.device)
            pred_Q1_t_choice = pred_Q1_t.gather(1, torch.tensor(actions).to(self.device).unsqueeze(-1)).squeeze(-1)
            pred_Q2_t = self.critic2.forward(states, target=True, device=self.device)
            pred_Q2_t_choice = pred_Q2_t.gather(1, torch.tensor(actions).to(self.device).unsqueeze(-1)).squeeze(-1)
            pred_Q_t = torch.minimum(pred_Q1_t_choice, pred_Q2_t_choice)

            # Calculate V(s') for each state in the batch
            pred_V_s_next = self.value.forward(next_states, device=self.device).squeeze(-1)
            pred_V_s_next = torch.where(~torch.tensor(dones).to(self.device), pred_V_s_next,
                                        torch.zeros_like(pred_V_s_next))

        # Calculate loss_qt
        target_q = torch.tensor(reward, dtype=torch.float32).to(self.device) + gamma * pred_V_s_next
        loss_qt = F.mse_loss(pred_Q_t, target_q)

        return loss_qt.item()

    def _validate_policy(self, val_data: list, beta):
        # Unpack the batch
        states, actions = zip(*[(exp.state, exp.action) for exp in val_data])

        # Calculate Qt(s,a) for each state in the batch
        with torch.no_grad():
            pred_Q1 = self.critic1.forward(states, target=True, device=self.device)
            pred_Q2 = self.critic2.forward(states, target=True, device=self.device)
            pred_Q = torch.minimum(pred_Q1, pred_Q2)
            pred_Q = pred_Q.gather(1, torch.tensor(actions).to(self.device).unsqueeze(-1)).squeeze(-1)

            # Calculate V(s) for each state in the batch
            pred_V_s = self.value.forward(states, device=self.device).squeeze(1)

            # Calculate the logprobs of the action taken
            action_logits = self.actor.forward(states, device=self.device)
            action_logprobs = torch.log_softmax(action_logits, dim=1)
            action_logprobs = action_logprobs.gather(1, torch.tensor(actions).to(self.device).unsqueeze(-1)).squeeze(-1)
            action_logprobs = torch.where(torch.isinf(action_logprobs), -1000, action_logprobs)
            action_logprobs = torch.where(action_logprobs == 0, -1e-8, action_logprobs)

        # Calculate Advantage filter
        advantage = pred_Q - pred_V_s

        loss_policy = action_logprobs * torch.exp(beta * advantage)
        loss_policy = -loss_policy.mean()

        return loss_policy.item()

    def _update_behaviour_policy(self, batch: list):
        """
        Variable 'batch' should be a 1D list of Experiences - sorted or unsorted. The reward value should be the
        correct Q_s value for that state i.e., the cumulated discounted reward from that state onwards.
        """

        # Unpack the batch
        states, actions = zip(*[(exp.state, exp.action) for exp in batch])

        # Calculate the logprobs of the action taken
        logits = self.behaviour_policy.forward(states, device=self.device)
        logprobs = torch.log_softmax(logits, dim=1)
        logprobs = logprobs.gather(1, torch.tensor(actions).to(self.device).unsqueeze(-1)).squeeze(-1)

        # Calculate the loss and backprop
        loss = -logprobs.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, episodes=100, parallel_envs=32, verbose=True, behaviour=False):

        @ray.remote
        def env_run(actor):
            done = False
            prem_done = False
            state = self.env.reset()[0]
            ep_record = []
            while not done and not prem_done:
                with torch.no_grad():
                    logits = actor.forward(state, device='cpu')

                # action = torch.argmax(logits).item()
                action_probs = F.softmax(logits, dim=-1)
                action_distribution = Categorical(action_probs)

                action = action_distribution.sample().item()

                next_state, reward, done, prem_done, _ = self.env.step(action)

                ep_record.append(Experience(state=state,
                                            action=action,
                                            reward=reward,
                                            next_state=next_state))

                state = next_state

            total_reward = sum([exp.reward for exp in ep_record])

            return ep_record, total_reward

        @ray.remote
        def env_run_behaviour(actor):
            done = False
            prem_done = False
            state = self.env.reset()[0]
            ep_record = []
            while not done and not prem_done:
                with torch.no_grad():
                    logits = actor.forward(state, device='cpu')

                action_probs = F.softmax(logits, dim=-1)
                action_distribution = Categorical(action_probs)

                action = action_distribution.sample().item()

                next_state, reward, done, prem_done, _ = self.env.step(action)

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

        print(f'Beginning evaluation over {episodes} episodes...\n') if verbose else None
        for episode in tqdm(range(episodes // parallel_envs), disable=not verbose, file=sys.stdout):
            if behaviour:
                temp_batch_records, temp_total_reward = zip(
                    *ray.get([env_run_behaviour.remote(self.behaviour_policy.cpu()) for _ in range(parallel_envs)]))
            else:
                temp_batch_records, temp_total_reward = zip(
                    *ray.get([env_run.remote(self.actor.cpu()) for _ in range(parallel_envs)]))

            self.actor.to(self.device)

            temp_batch_states, temp_batch_actions, temp_batch_rewards, temp_batch_next_states = zip(
                *[(exp.state, exp.action, exp.reward, exp.next_state)
                  for episode in temp_batch_records for exp in episode])

            all_states += list(temp_batch_states)
            all_actions += list(temp_batch_actions)
            all_rewards += list(temp_batch_rewards)
            all_next_states += list(temp_batch_next_states)
            all_total_rewards += list(temp_total_reward)

        print(
            f'Evaluation complete! Average reward per episode: {np.array(all_total_rewards).mean()}') if verbose else None

        return np.array(all_states), np.array(all_actions), np.array(all_rewards), \
            np.array(all_next_states), np.array(all_total_rewards)

    def evaluate_offline(self, data, episodes=100, parallel_envs=32, verbose=True):

        dones = np.array([exp.done for exp in data])
        episode_idxs = np.where(dones)[0] + 1
        episode_idxs = episode_idxs.tolist()
        episode_idxs.insert(0, 0)
        episode_idxs = np.array(episode_idxs)
        episode_idxs = [slice(i, j) for i, j in zip(episode_idxs[:-1], episode_idxs[1:])]
        np.random.default_rng().shuffle(episode_idxs)

        for ep_idx in episode_idxs[:episodes]:
            p = 1.0
            h = []
            t = 0
            r = 0
            for exp in data[ep_idx]:
                h.append(exp.state)
                logits = self.actor.forward(exp.state, device='cpu')
                actionprobs = F.softmax(logits)[exp.action]
                p *= actionprobs

        # return np.array(all_states), np.array(all_actions), np.array(all_rewards), \
        # np.array(all_next_states), np.array(all_total_rewards)
        return

    def choose_action(self, state, device='cpu'):
        with torch.no_grad():
            logits = self.actor.forward(state, device=device)

        action = torch.argmax(logits)

        return action.item()

    def load_model(self, criticpath1=None, criticpath2=None, valuepath=None, actorpath=None, behaviourpolicypath=None):
        if criticpath1 is not None:
            self.critic1.target_net = torch.load(criticpath1)
            self.critic1.has_net = True
            self.path = os.path.dirname(criticpath1)
        if criticpath2 is not None:
            self.critic2.target_net = torch.load(criticpath2)
            self.critic2.has_net = True
        if valuepath is not None:
            self.value = torch.load(valuepath)
            self.value.has_net = True
            self.path = os.path.dirname(valuepath)
        if actorpath is not None:
            self.actor = torch.load(actorpath)
            self.actor.has_net = True
            self.path = os.path.dirname(actorpath)
        if behaviourpolicypath is not None:
            self.behaviour_policy = torch.load(behaviourpolicypath)
            self.behaviour_policy.has_net = True

        if self._compiled:
            self._compiled = False
            self._optimizer = None
            self._learning_rate = None
            self._regularisation = None
            print('Model loaded - recompile agent please')


class IQLVariableAgent(BaseAgent):
    """
    IQLVariableAgent is an expansion of IQLAgent, allowing for variable lengths between decisions.
    """

    def __init__(self, device):
        super().__init__(device)
        self._alpha = None
        self.custom_params = None
        self._beta = None
        self._tau = None
        self.value = CriticNet()
        self.critic = DualNet()
        self.actor = ActorNet()
        self.behaviour_policy = ActorNet()
        self.net = self.critic1
        self._batch_size = None

    def reset(self):
        reset_done = super().reset()

        if reset_done:
            self._alpha = None
            self.custom_params = None
            self._beta = None
            self._tau = None
            self.value = CriticNet()
            self.critic = DualNet()
            self.actor = ActorNet()
            self.behaviour_policy = ActorNet()
            self.net = self.actor
            self._batch_size = None

    def add_network(self, nodes: list):
        super().network_check(nodes)

        self.value.add_layers(nodes, self.env)
        self.critic.add_layers(nodes, self.env)
        self.actor.add_layers(nodes, self.env)
        self.behaviour_policy.add_layers(nodes, self.env)
        self.value.has_net = True
        self.critic.has_net = True
        self.actor.has_net = True
        self.behaviour_policy.has_net = True

    def compile(self, optimizer, learning_rate=0.001, weight_decay=1e-5, clip=1.0, lower_clip=None, upper_clip=None):
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._regularisation = weight_decay
        super().compile_check()

        self.value.to(self.device)
        self.critic.main_net.to(self.device)
        self.critic.target_net.to(self.device)
        self.actor.to(self.device)
        self.behaviour_policy.to(self.device)
        """
        for params in [self.value.parameters(), self.critic1.main_net.parameters(), self.critic2.main_net.parameters(),
                       self.critic1.target_net.parameters(), self.critic2.target_net.parameters(),
                       self.actor.parameters(), self.behaviour_policy.parameters()]:
        """
        for params in [self.value.parameters(), self.critic.main_net.parameters(), self.critic.target_net.parameters(),
                       self.actor.parameters(),
                       self.behaviour_policy.parameters()]:
            for p in params:
                p.register_hook(lambda grad: torch.clamp(grad,
                                                         lower_clip if lower_clip is not None else -clip,
                                                         upper_clip if upper_clip is not None else clip))

        self._compiled = True

    @staticmethod
    def sample(data, batch_size):
        idxs = np.random.default_rng().choice(len(data), size=batch_size, replace=False)
        return [data[idx] for idx in idxs]

    def clone_behaviour(self, data, batch_size=64, epochs=10000, evaluate=False, save=False):

        """
        Draft - need to check if any changes are required to make this actually function correctly.
        """

        # Set up optimiser
        self.custom_params = []
        self.custom_params.append({'params': self.behaviour_policy.parameters(),
                                   'lr': self._learning_rate,
                                   'weight_decay': self._regularisation})

        super().train_base(0.99, custom_params=self.custom_params)

        # Make save directory if needed
        if save:
            if not os.path.isdir(self.path):
                os.makedirs(self.path)

        # Create logs
        old_policy_loss = torch.inf
        best_policy_loss = torch.inf
        current_loss_policy = []

        # Start training
        print('Beginning training...\n')
        progress_bar = tqdm(range(1, int(epochs) + 1), file=sys.stdout)
        for epoch in progress_bar:
            batch = self.sample(data, batch_size)

            loss = self._update_behaviour_policy(batch)

            current_loss_policy.append(loss)

            if epoch % 1 == 0:
                if evaluate:
                    _, _, _, _, total_rewards = self.evaluate(episodes=1000, parallel_envs=512,
                                                              verbose=False, behaviour=True)

                    with open(
                            '/Users/thomasfrost/Documents/Github/pygment/Informal experiments/clone_behaviour/all_rewards.txt',
                            'a') as f:
                        f.write(f'{int(np.array(total_rewards.mean()))} ')
                        f.close()

                    with open(
                            '/Users/thomasfrost/Documents/Github/pygment/Informal experiments/clone_behaviour/policy_loss.txt',
                            'a') as f:
                        f.write(f'{round(np.array(current_loss_policy).mean(), 6)} ')
                        f.close()

                print(f'\nSteps completed: {epoch}\n')
                print(
                    f'Behaviour policy loss {"decreased" if np.array(current_loss_policy).mean() < old_policy_loss else "increased"} '
                    f'from {old_policy_loss} to {round(np.array(current_loss_policy).mean(), 5)}'
                )
                print(f'Best policy loss: {min(best_policy_loss, round(np.array(current_loss_policy).mean(), 5))}')
                if evaluate:
                    print(f'Average reward: {int(np.array(total_rewards).mean())}')

                if np.array(current_loss_policy).mean() < best_policy_loss:
                    if save:
                        for net, name in [
                            [self.behaviour_policy, 'behaviour_policy']
                        ]:
                            old_save_path = os.path.join(self.path, f'{name}_{best_policy_loss}.pt')
                            new_save_path = os.path.join(self.path,
                                                         f'{name}_{round(np.array(current_loss_policy).mean(), 5)}.pt')

                            torch.save(net, new_save_path)
                            if os.path.isfile(old_save_path) and old_save_path != new_save_path:
                                os.remove(old_save_path)

                    best_policy_loss = round(np.array(current_loss_policy).mean(), 5)

                old_policy_loss = round(np.array(current_loss_policy).mean(), 5)
                current_loss_policy = []

    def train(self, data, critic=True, value=True, actor=True, evaluate=True, steps=1000, batch_size=64,
              gamma=0.99, tau=0.99, beta=1, update_iter=4, ppo_clip=0.01, ppo_clip_decay=0.9, save=False):
        """
        Variable 'data' should be a 1D list of Experiences - sorted or unsorted. The reward value should be the
        correct Q_s value for that state i.e., the cumulated discounted reward from that state onwards.
        """
        if save:
            evaluate = True

        # Set up optimiser
        self.method_check(env_loaded=True, net_exists=True, compiled=True)
        self._gamma = gamma

        self.optimizer = {}
        scheduler = {}

        for network_name, network in [
            ['value', self.value],
            ['critic_main', self.critic.main_net],
            ['critic_target', self.critic.target_net],
            ['actor', self.actor]
        ]:
            self.optimizer[network_name] = self._optimizers[self._optimizer](network.parameters(),
                                                                             lr=self._learning_rate,
                                                                             weight_decay=self._regularisation)

            scheduler[network_name] = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer[network_name],
                                                                                 T_max=steps)

        # Create stochastic weight averaged models
        swa_critic_main_net = torch.optim.swa_utils.AveragedModel(self.critic.main_net)
        swa_critic_target_net = torch.optim.swa_utils.AveragedModel(self.critic.target_net)
        swa_value = torch.optim.swa_utils.AveragedModel(self.value)

        swa_scheduler = {}
        for optim_name in ['critic_main', 'critic_target', 'value']:
            swa_scheduler[optim_name] = torch.optim.swa_utils.SWALR(self.optimizer[optim_name], swa_lr=0.01)

        # Make save directory if needed
        if save:
            if not os.path.isdir(self.path):
                os.makedirs(self.path)

        # Create logs
        old_qt_loss = torch.inf
        old_v_loss = torch.inf
        old_policy_loss = torch.inf
        old_average_reward = -10 ** 10
        current_loss_qt = []
        current_loss_v = []
        current_loss_policy = []
        current_best_reward = -10 ** 10

        # If evaluating, start ray instance
        # if evaluate:
        # ray.init()

        # Start training
        print('Beginning training...\n')
        progress_bar = tqdm(range(1, int(steps) + 1), file=sys.stdout)
        for step in progress_bar:
            batch = self.sample(data, batch_size)

            loss_qt = self._update_q(batch, gamma) if critic else None
            loss_v = self._update_v(batch, tau) if value else None

            current_loss_qt.append(loss_qt)
            current_loss_v.append(loss_v)

            if step > 50 and step % 5 == 0:
                swa_critic_main_net.update_parameters(self.critic.main_net)
                swa_critic_target_net.update_parameters(self.critic.target_net)
                swa_value.update_parameters(self.value)
                for optim_name in ['critic_main', 'critic_target', 'value']:
                    swa_scheduler[optim_name].step()

                loss_policy = self._update_policy(batch, beta, update_iter, ppo_clip) if actor else None
                current_loss_policy.append(loss_policy)
                ppo_clip *= ppo_clip_decay

            else:
                for network_name in ['value', 'critic_main', 'critic_target', 'actor']:
                    scheduler[network_name].step()

            if step % 100 == 0:

                if evaluate:
                    _, _, _, _, total_rewards = self.evaluate(episodes=1000, parallel_envs=512,
                                                              verbose=False)

                    with open(
                            '/Users/thomasfrost/Documents/Github/pygment/Informal experiments/mse_loss/all_rewards.txt',
                            'a') as f:
                        f.write(f'{int(np.array(total_rewards.mean()))} ')
                        f.close()

                else:
                    total_rewards = np.array([0])

                print(f'\nSteps completed: {step}\n')
                if critic:
                    print(
                        f'Qt_loss {"decreased" if np.array(current_loss_qt).mean() < old_qt_loss else "increased"} '
                        f'from {round(old_qt_loss, 5)} to {round(np.array(current_loss_qt).mean(), 5)}'
                    )
                if value:
                    print(
                        f'V_loss {"decreased" if np.array(current_loss_v).mean() < old_v_loss else "increased"} '
                        f'from {round(old_v_loss, 5)} to {round(np.array(current_loss_v).mean(), 5)}'
                    )
                if actor:
                    print(
                        f'Policy loss {"decreased" if np.array(current_loss_policy).mean() < old_policy_loss else "increased"} '
                        f'from {round(old_policy_loss, 5)} to {round(np.array(current_loss_policy).mean(), 5)}'
                    )
                if evaluate:
                    print(
                        f'Average reward {"decreased" if total_rewards.mean() < old_average_reward else "increased"} '
                        f'from {int(old_average_reward)} to {int(total_rewards.mean())}'
                    )
                    print(
                        f'Best reward {max(current_best_reward, int(total_rewards.mean()))}'
                    )

                if total_rewards.mean() > current_best_reward:
                    if save:
                        for net, name in [
                            [self.critic.main_net, 'critic_main'],
                            [self.critic.target_net, 'critic_target'],
                            [self.value, 'value'],
                            [self.actor, 'actor']
                        ]:
                            old_save_path = os.path.join(self.path, f'{name}_{int(current_best_reward)}.pt')
                            new_save_path = os.path.join(self.path, f'{name}_{int(total_rewards.mean())}.pt')

                            torch.save(net, new_save_path)
                            if os.path.isfile(old_save_path) and old_save_path != new_save_path:
                                os.remove(old_save_path)

    def _update_q(self, batch: list, gamma):
        """
        Variable 'batch' should be a 1D list of Experiences - sorted or unsorted. The reward value should be the
        correct Q_s value for that state i.e., the cumulated discounted reward from that state onwards.
        """

        # Calculate Q loss
        loss_qt = calc_iql_q_loss_batch(batch, self.device, self.critic, self.value, gamma)

        # Update Networks
        for network_name in ['critic_main', 'critic_target']:
            self.optimizer[network_name].zero_grad()
        loss_qt.backward()
        for network_name in ['critic_main', 'critic_target']:
            self.optimizer[network_name].step()

        return loss_qt.item()

    def _update_v(self, batch: list, tau):
        """
        Variable 'batch' should be a 1D list of Experiences - sorted or unsorted. The reward value should be the
        correct Q_s value for that state i.e., the cumulated discounted reward from that state onwards.
        """

        # Calculate V loss
        loss_v = calc_iql_v_loss_batch(batch, self.device, self.value, tau)

        # Update Networks
        self.optimizer['value'].zero_grad()
        loss_v.backward()
        self.optimizer['value'].step()

        return loss_v.item()

    def _update_policy(self, batch: list, beta=1, update_iter=4, ppo_clip=0.01):
        """
        Variable 'batch' should be a 1D list of Experiences - sorted or unsorted. The reward value should be the
        correct Q_s value for that state i.e., the cumulated discounted reward from that state onwards.
        """

        loss_policy = calc_iql_policy_loss_batch(batch, self.device, self.critic, self.value,
                                                 self.actor)

        self.optimizer['actor'].zero_grad()
        loss_policy.backward()
        self.optimizer['actor'].step()

        return loss_policy.item()

        """
        # Unpack the batch
        states, actions = zip(*[(exp.state, exp.action) for exp in batch])

        # Calculate the logprobs of the action taken
        with torch.no_grad():
            old_action_logits = self.actor.forward(states, device=self.device)
            old_action_logprobs = torch.log_softmax(old_action_logits, dim=1)
            old_action_logprobs = old_action_logprobs.gather(1,
                                                             torch.tensor(actions).to(self.device).unsqueeze(
                                                                 -1)).squeeze(
                -1)
            old_action_logprobs = torch.where(torch.isinf(old_action_logprobs), -1000, old_action_logprobs)
            old_action_logprobs = torch.where(old_action_logprobs == 0, -1e-8, old_action_logprobs)

        total_loss_policy = 0

        for _ in range(update_iter):
            loss_policy = calc_iql_policy_loss_batch(batch, self.device, self.critic1, self.critic2,
                                                     self.value, self.actor, old_action_logprobs, beta,
                                                     ppo_clip)

            self.optimizer['actor'].zero_grad()
            loss_policy.backward()
            self.optimizer['actor'].step()

            total_loss_policy += loss_policy.item()

        return total_loss_policy
        """

    def _update_behaviour_policy(self, batch: list):
        """
        Variable 'batch' should be a 1D list of Experiences - sorted or unsorted. The reward value should be the
        correct Q_s value for that state i.e., the cumulated discounted reward from that state onwards.
        """

        # Unpack the batch
        states, actions = zip(*[(exp.state, exp.action) for exp in batch])

        # Calculate the logprobs of the action taken
        logits = self.behaviour_policy.forward(states, device=self.device)
        logprobs = torch.log_softmax(logits, dim=1)
        logprobs = logprobs.gather(1, torch.tensor(actions).to(self.device).unsqueeze(-1)).squeeze(-1)

        # Calculate the loss and backprop
        loss = -logprobs.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, episodes=100, parallel_envs=32, verbose=True, behaviour=False):

        @ray.remote
        def env_run(actor):
            done = False
            prem_done = False
            state = self.env.reset()[0]
            ep_record = []
            while not done and not prem_done:
                with torch.no_grad():
                    logits = actor.forward(state, device='cpu')

                # action = torch.argmax(logits).item()
                action_probs = F.softmax(logits, dim=-1)
                action_distribution = Categorical(action_probs)

                action = action_distribution.sample().item()

                next_state, reward, done, prem_done, _ = self.env.step(action)

                ep_record.append(Experience(state=state,
                                            action=action,
                                            reward=reward,
                                            next_state=next_state))

                state = next_state

            total_reward = sum([exp.reward for exp in ep_record])

            return ep_record, total_reward

        @ray.remote
        def env_run_behaviour(actor):
            done = False
            prem_done = False
            state = self.env.reset()[0]
            ep_record = []
            while not done and not prem_done:
                with torch.no_grad():
                    logits = actor.forward(state, device='cpu')

                action_probs = F.softmax(logits, dim=-1)
                action_distribution = Categorical(action_probs)

                action = action_distribution.sample().item()

                next_state, reward, done, prem_done, _ = self.env.step(action)

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

        print(f'Beginning evaluation over {episodes} episodes...\n') if verbose else None
        for episode in tqdm(range(episodes // parallel_envs), disable=not verbose, file=sys.stdout):
            if behaviour:
                temp_batch_records, temp_total_reward = zip(
                    *ray.get([env_run_behaviour.remote(self.behaviour_policy.cpu()) for _ in range(parallel_envs)]))
            else:
                temp_batch_records, temp_total_reward = zip(
                    *ray.get([env_run.remote(self.actor.cpu()) for _ in range(parallel_envs)]))

            self.actor.to(self.device)

            temp_batch_states, temp_batch_actions, temp_batch_rewards, temp_batch_next_states = zip(
                *[(exp.state, exp.action, exp.reward, exp.next_state)
                  for episode in temp_batch_records for exp in episode])

            all_states += list(temp_batch_states)
            all_actions += list(temp_batch_actions)
            all_rewards += list(temp_batch_rewards)
            all_next_states += list(temp_batch_next_states)
            all_total_rewards += list(temp_total_reward)

        print(
            f'Evaluation complete! Average reward per episode: {np.array(all_total_rewards).mean()}') if verbose else None

        return np.array(all_states), np.array(all_actions), np.array(all_rewards), \
            np.array(all_next_states), np.array(all_total_rewards)

    def evaluate_offline(self, data, episodes=100, parallel_envs=32, verbose=True):

        dones = np.array([exp.done for exp in data])
        episode_idxs = np.where(dones)[0] + 1
        episode_idxs = episode_idxs.tolist()
        episode_idxs.insert(0, 0)
        episode_idxs = np.array(episode_idxs)
        episode_idxs = [slice(i, j) for i, j in zip(episode_idxs[:-1], episode_idxs[1:])]
        np.random.default_rng().shuffle(episode_idxs)

        for ep_idx in episode_idxs[:episodes]:
            p = 1.0
            h = []
            t = 0
            r = 0
            for exp in data[ep_idx]:
                h.append(exp.state)
                logits = self.actor.forward(exp.state, device='cpu')
                actionprobs = F.softmax(logits)[exp.action]
                p *= actionprobs

        # return np.array(all_states), np.array(all_actions), np.array(all_rewards), \
        # np.array(all_next_states), np.array(all_total_rewards)
        return

    def choose_action(self, state, device='cpu'):
        with torch.no_grad():
            logits = self.actor.forward(state, device=device)

        action = torch.argmax(logits)

        return action.item()

    def load_model(self, criticpath1=None, criticpath2=None, valuepath=None, actorpath=None, behaviourpolicypath=None):
        if criticpath1 is not None:
            if '_main' in criticpath1:
                self.critic1.main_net = torch.load(criticpath1)
                self.critic1.target_net = torch.load(criticpath1.replace('_main', '_target'))
            else:
                self.critic1.main_net = torch.load(criticpath1.replace('_target', '_main'))
                self.critic1.target_net = torch.load(criticpath1)
            self.critic1.has_net = True
            self.path = os.path.dirname(criticpath1)
        if criticpath2 is not None:
            if '_main' in criticpath1:
                self.critic2.main_net = torch.load(criticpath2)
                self.critic2.target_net = torch.load(criticpath2.replace('_main', '_target'))
            else:
                self.critic2.main_net = torch.load(criticpath2.replace('_target', '_main'))
                self.critic2.target_net = torch.load(criticpath2)
            self.critic2.has_net = True
        if valuepath is not None:
            self.value = torch.load(valuepath)
            self.value.has_net = True
        if actorpath is not None:
            self.actor = torch.load(actorpath)
            self.actor.has_net = True
        if behaviourpolicypath is not None:
            self.behaviour_policy = torch.load(behaviourpolicypath)
            self.behaviour_policy.has_net = True

        if self._compiled:
            self._compiled = False
            self._optimizer = None
            self._learning_rate = None
            self._regularisation = None
            print('Model loaded - recompile agent please')


class PolicyGradient(BaseAgent):
    """
    PolicyGradient is a neural network based on REINFORCE
    """

    def __init__(self, device, path):
        super().__init__(device, path)
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

    def __init__(self, device, path):
        super().__init__(device, path)
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

    def train(self, target_reward, save_from, save_interval=10, iterations=10000, sample_episodes=100, parallel_envs=32,
              update_iter=4, update_steps=1000, batch_size=128, gamma=0.99):
        super().train_base(gamma)

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

        for iteration in range(1, iterations):
            batch_Q_s = []
            batch_states = []
            batch_actions = []
            batch_old_policy_logprobs = []
            while True:
                temp_batch_records, temp_batch_Q_s = [], []
                for _ in range(ceil(sample_episodes / parallel_envs)):
                    current_batch_records, current_batch_Q_s = zip(
                        *ray.get([env_run.remote(self.net.cpu()) for _ in range(parallel_envs)]))

                    temp_batch_records += list(current_batch_records)
                    temp_batch_Q_s += list(current_batch_Q_s)

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
                f'Epoch: {iteration}, '
                f'Loss {round(np.array(total_loss).mean(), 2)} '
                f'(Policy {round(np.array(total_policy_loss).mean(), 2)}, '
                f'Value {round(np.array(total_value_loss).mean(), 2)}), '
                f'Mean Reward: {round(np.array(total_rewards).mean(), 2)}')

    def explore(self, episodes=10000, parallel_envs=32, gamma=0.99):
        super().train_base(gamma)
        if not ray.is_initialized():
            ray.init()

        @ray.remote
        def env_run(predict_net):
            done = False
            prem_done = False
            ep_record = []

            state = self.env.reset()[0]
            with torch.no_grad():
                action_logits, _ = predict_net.forward(state, device='cpu')

            action, _, _ = self.get_action_and_logprobs(action_logits)

            action = action.item()

            while not done and not prem_done:

                next_state, reward, done, prem_done, _ = self.env.step(action)

                if not done and not prem_done:
                    with torch.no_grad():
                        next_action_logits, _ = predict_net.forward(next_state, device='cpu')

                    next_action, _, _ = self.get_action_and_logprobs(next_action_logits)

                    next_action = next_action.item()

                ep_record.append(Experience(state=state,
                                            action=action,
                                            reward=reward,
                                            next_state=next_state,
                                            next_action=next_action,
                                            done=done))

                state = next_state
                action = next_action

            cum_rewards = calc_cum_rewards([exp.reward for exp in ep_record], self._gamma)

            return ep_record, cum_rewards

        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_next_actions = []
        all_dones = []
        all_cum_rewards = []

        print(f'\nBeginning exploration over {episodes} episodes...')
        for episode in tqdm(range(episodes // parallel_envs), file=sys.stdout):
            temp_batch_records, temp_batch_Qs = zip(
                *ray.get([env_run.remote(self.net.cpu()) for _ in range(parallel_envs)]))

            temp_batch_states, temp_batch_actions, temp_batch_rewards, temp_batch_next_states, \
                temp_next_actions, temp_batch_dones = zip(*[(exp.state, exp.action, exp.reward, exp.next_state,
                                                             exp.next_action, exp.done)
                                                            for episode in temp_batch_records for exp in episode])

            all_states += list(temp_batch_states)
            all_actions += list(temp_batch_actions)
            all_rewards += list(temp_batch_rewards)
            all_next_states += list(temp_batch_next_states)
            all_next_actions += list(temp_next_actions)
            all_dones += list(temp_batch_dones)
            all_cum_rewards += list([Q_s for episode in temp_batch_Qs for Q_s in episode])

        return np.array(all_states), np.array(all_actions), np.array(all_rewards), \
            np.array(all_next_states), np.array(all_next_actions), np.array(all_dones), np.array(all_cum_rewards)


class PPOContinuous(BaseAgent):
    """
  PPOContinuous uses the actor-critic neural network to solve environments with a continuous action space.
  """

    def __init__(self, device, path):
        super().__init__(device, path)
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
        super().train_base(gamma, custom_params)

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
