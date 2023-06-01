import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
from copy import deepcopy
import numpy as np


class BaseNet:
    """
    Base neural network
    """

    def __init__(self):
        super().__init__()
        self.has_net = False
        self.action_space = None
        self.observation_space = None

        self.activations = {'relu': nn.ReLU,
                            'sigmoid': nn.Sigmoid,
                            'leaky': nn.LeakyReLU}

        self.output_activations = {'sigmoid': nn.Sigmoid,
                                   'softmax': nn.Softmax,
                                   'linear': None}

    def env_is_discrete(self, env):
        if type(env.action_space) == gym.spaces.discrete.Discrete:
            return True

        elif type(env.action_space) == gym.spaces.box.Box:
            return False

        else:
            raise TypeError('Environment action space is neither Box nor Discrete...')

    def add_base_layers(self, nodes: list, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

        input_layers = nn.ModuleList([])

        for layer in range(len(nodes)):
            if layer == 0:
                input_layers.append(nn.Linear(self.observation_space, nodes[layer]))
                input_layers.append(nn.ReLU())
            else:
                input_layers.append(nn.Linear(nodes[layer - 1], nodes[layer]))
                input_layers.append(nn.ReLU())

        input_layers.append(nn.Linear(nodes[-1], self.action_space))

        return input_layers


class DualNet(BaseNet):
    """
    Wrapper around model to create both a main_net and a target_net
    """

    def __init__(self, has_dual=True):
        super().__init__()
        self.main_net = None
        self.target_net = None
        self.has_dual = has_dual
        self.sync_tracker = 0

    def sync(self, tau):
        if not self.has_dual:
            return None
        if isinstance(tau, int):
            if self.sync_tracker == tau:
                self.target_net.load_state_dict(self.main_net.state_dict())
                self.sync_tracker = 0
            else:
                self.sync_tracker += 1

        else:
            for target_param, main_param in zip(self.target_net.parameters(), self.main_net.parameters()):
                target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

    def add_layers(self, nodes: list, env):
        if not self.env_is_discrete(env):
            raise TypeError('Action space is continuous, not discrete - please use a continuous policy')

        self.main_net = super().add_base_layers(nodes, env.observation_space.shape[0], env.action_space.n)
        if self.has_dual:
            self.target_net = deepcopy(self.main_net)

    def forward(self, state, target=False, device='cpu'):
        if not self.has_dual:
            target = False
        net = self.target_net if target else self.main_net

        Q_s = torch.tensor(state).to(device)

        for layer in net:
            Q_s = layer(Q_s)

        return Q_s


class PolicyGradientNet(BaseNet, nn.Module):
    """
    Wrapper for a policy gradient-based neural network
    """

    def __init__(self):
        super().__init__()
        self.net = None

    def add_layers(self, nodes: list, env):
        if not self.env_is_discrete(env):
            raise TypeError('Action space is continuous, not discrete - please use a continuous policy')

        self.net = super().add_base_layers(nodes, env.observation_space.shape[0], env.action_space.n)

    def forward(self, state, device='cpu'):
        action_logits = torch.tensor(state).to(device)

        for layer in self.net:
            action_logits = layer(action_logits)

        action_probs = F.softmax(action_logits, dim=-1)
        action_logprobs = F.log_softmax(action_logits, dim=-1)
        action_distribution = Categorical(action_probs)
        # Following is to avoid rare events where probability is represented as zero (and logprob = inf),
        # but is in fact non-zero, and an action is sampled from this index.
        while True:
            action = action_distribution.sample()
            if action.shape:
                if ~torch.isinf(action_logprobs.gather(1, action.unsqueeze(-1)).squeeze(-1)).all():
                    break
            else:
                if ~torch.isinf(action_logprobs[action.item()]):
                    break

        return action, action_probs, action_logprobs


class ActorCriticNet(BaseNet, nn.Module):
    """
    Wrapper for the Actor-Critic neural networks
    """

    def __init__(self):
        super().__init__()
        self.base_net = None
        self.actor_net = None
        self.critic_net = None

    def add_layers(self, nodes: list, env):
        if not self.env_is_discrete(env):
            raise TypeError('Action space is continuous, not discrete - please use a continuous policy')

        self.actor_net = super().add_base_layers(nodes, env.observation_space.shape[0], env.action_space.n)
        self.critic_net = super().add_base_layers(nodes, env.observation_space.shape[0], 1)

    def forward(self, state, device='cpu'):
        state_value = torch.tensor(np.array(state)).to(device)
        action_logits = torch.tensor(np.array(state)).to(device)

        for layer_idx in range(len(self.actor_net)):
            state_value = self.critic_net[layer_idx](state_value)
            action_logits = self.actor_net[layer_idx](action_logits)

        return action_logits, state_value


class ActorCriticNetContinuous(BaseNet, nn.Module):
    """
  Wrapper for the Actor-Critic neural networks (for a continuous action space)
  """

    def __init__(self):
        super().__init__()
        self.critic_net = None
        self.actor_net = None

        self.clip_high = None
        self.clip_low = None

    def add_layers(self, nodes: list, env):
        if self.env_is_discrete(env):
            raise TypeError('Action space is discrete, not continuous - please use a discrete policy')

        self.clip_high = torch.tensor(env.action_space.high)
        self.clip_low = torch.tensor(env.action_space.low)

        self.critic_net = super().add_base_layers(nodes, env.observation_space.shape[0], 1)
        self.actor_net = super().add_base_layers(nodes, env.observation_space.shape[0], env.action_space.shape[0] * 2)
        # self.actor_net = super().add_base_layers(nodes, env.observation_space.shape[0], 1)[:-1]
        # self.actor_net.append(nn.ModuleDict())
        # self.actor_net[-1]['mu'] = nn.ModuleList([nn.Linear(nodes[-1], env.action_space.shape[0])])
        # self.actor_net[-1]['sigma'] = nn.ParameterList([nn.Parameter(torch.ones(env.action_space.shape[0])*0.5,
        # requires_grad=True)])

        # Change from ReLU to Tanh
        '''for idx in [i for i in range(len(nodes)*2) if i % 2 != 0]:
        self.critic_net[idx] = nn.Tanh()
        self.actor_net[idx] = nn.Tanh()'''

    def forward(self, state, device='cpu'):
        state_value = torch.tensor(state).to(device)
        action_means = torch.tensor(state).to(device)

        for layer_idx in range(len(self.actor_net)):
            state_value = self.critic_net[layer_idx](state_value)
            action_means = self.actor_net[layer_idx](action_means)
            # if layer_idx < len(self.actor_net)-1:
            # action_means = self.actor_net[layer_idx](action_means)
            # else:
            # action_means = self.actor_net[layer_idx]['mu'][0](action_means)
            # action_stds = self.actor_net[layer_idx]['sigma'][0] + 1e-8

        action_means = action_means.reshape(-1, action_means.shape[-1])

        action_stds = torch.clip(torch.nn.Softplus()(action_means[:, action_means.shape[-1] // 2:]),
                                 min=1e-8)
        action_means = action_means[:, :action_means.shape[-1] // 2]

        return action_means, action_stds, state_value
