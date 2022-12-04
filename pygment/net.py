import gymnasium as gym
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from copy import deepcopy


class BaseNet:
    """
    Base neural network
    """
    def __init__(self):
        super().__init__()
        self.has_net = False
        self.observation_space = None
        self.action_space = None

        self.activations = {'relu': nn.ReLU,
                            'sigmoid': nn.Sigmoid,
                            'leaky': nn.LeakyReLU}

        self.output_activations = {'sigmoid': nn.Sigmoid,
                                   'softmax': nn.Softmax,
                                   'linear': None}


    def add_layers(self, nodes: list, env):
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        if type(self.action_space) == gym.spaces.box.Box:
            if len(self.action_space.shape) > 1:
                raise TypeError('Environment action space is multidimensional...')
            self._is_continuous = True

        elif type(self.action_space) == gym.spaces.discrete.Discrete:
            self._is_continuous = False
        else:
            raise TypeError('Environment action space is neither Box nor Discrete...')

        input_layers = nn.ModuleList([])

        for layer in range(len(nodes)):
            if layer == 0:
                input_layers.append(nn.Linear(self.observation_space.shape[0], nodes[layer]))
                input_layers.append(nn.ReLU())
            else:
                input_layers.append(nn.Linear(nodes[layer - 1], nodes[layer]))
                input_layers.append(nn.ReLU())

        if self._is_continuous:
            input_layers.append(nn.Linear(nodes[-1], self.action_space.shape[0]))
            mu = deepcopy(input_layers)
            mu.append(nn.Tanh())
            sigma = deepcopy(input_layers)
            sigma.append(nn.Softplus())

            return {'mu': mu, 'sigma': sigma}

        return input_layers


class DualNet(BaseNet):
    """
    Wrapper around model to create both a main_net and a target_net
    """

    def __init__(self):
        super().__init__()
        self.main_net = None
        self.target_net = None
        self.sync_tracker = 0


    def sync(self, tau):
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
        if type(env.action_space) != gym.spaces.discrete.Discrete:
            raise TypeError('Pygment is not currently equipped to compute continuous DQN - please choose a policy-based method')
        self.main_net = super().add_layers(nodes, env)


    def forward(self, state, target=False, device='mps'):
        net = self.target_net if target else self.main_net

        state = torch.tensor(state).to(device)

        for layer in net[:-1]:
            state = F.relu(layer(state))

        Q_s = net[-1](state)

        return Q_s


class PolicyGradientNet(BaseNet, nn.Module):
    """
    Wrapper for a policy gradient-based neural network
    """

    def __init__(self):
        super().__init__()
        self.net = None


    def add_layers(self, nodes: list, env):
        self.net = super().add_layers(nodes, env)


    def forward(self, state, device='mps'):
        state = torch.tensor(state).to(device)

        for layer in self.net[:-1]:
            state = F.relu(layer(state))

        action_logits = self.net[-1](state)
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
        self.actor_net = None
        self.critic_net = None


    def add_layers(self, nodes: list, env):
        self.actor_net = super().add_layers(nodes, env)
        self.critic_net = super().add_layers(nodes, env)[:-1]
        self.critic_net.append(nn.Linear(nodes[-1], 1))


    def forward(self, state, device='mps'):
        state_value = torch.tensor(state).to(device)

        for layer in self.critic_net[:-1]:
          state_value = F.relu(layer(state_value))

        state_value = self.critic_net[-1](state_value)

        if self._is_continuous:
            return self.fwd_continuous(state, device), state_value

        else:
            return self.fwd_discrete(state, device), state_value


    def fwd_continuous(self, state, device='mps'):
        action_means_stds = torch.tensor(state).to(device)

        for layer in self.actor_net[:-1]:
            action_means_stds = F.relu(layer(action_means_stds))

        action_means_stds = self.actor_net[-1](action_means_stds)
        action_means = action_means_stds[:self.action_space.shape[0]]
        action_stds = torch.nn.Softplus(action_means_stds[self.action_space.shape[0]:])

        action = torch.normal(action_means,
                              action_stds)

        action_logprobs = -(action - action_means)**2 / (2 * action_stds ** 2)
        action_logprobs -= torch.log(torch.sqrt(2 * torch.pi * action_stds ** 2))

        entropy = torch.log(torch.sqrt(2 * torch.pi * torch.exp(torch.tensor(1)) * action_stds ** 2))

        return action, entropy, action_logprobs


    def fwd_discrete(self, state, device='mps'):
        action_logits = torch.tensor(state).to(device)

        for layer in self.actor_net[:-1]:
          action_logits = F.relu(layer(action_logits))

        action_logits = self.actor_net[-1](action_logits)

        # Actor layer:
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

        entropy = (batch_action_probs * batch_action_logprobs).sum(1).mean()

        return action, entropy, action_logprobs


