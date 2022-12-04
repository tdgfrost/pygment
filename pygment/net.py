import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym


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


    def add_layers(self, nodes: list, observation_space, action_space):
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
        if not self.env_is_discrete(env):
            raise TypeError('Action space is continuous, not discrete - please use a continuous policy')

        self.main_net = super().add_layers(nodes, env.observation_space.shape[0], env.action_space.n)


    def forward(self, state, target=False, device='mps'):
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

      self.net = super().add_layers(nodes, env.observation_space.shape[0], env.action_space.n)


    def forward(self, state, device='mps'):
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
        self.actor_net = None
        self.critic_net = None


    def add_layers(self, nodes: list, env):
        if not self.env_is_discrete(env):
            raise TypeError('Action space is continuous, not discrete - please use a continuous policy')

        self.actor_net = super().add_layers(nodes, env.observation_space.shape[0], env.action_space.n)
        self.critic_net = super().add_layers(nodes, env.observation_space.shape[0], 1)



    def forward(self, state, device='mps'):
        state_value = torch.tensor(state).to(device)
        action_logits = torch.tensor(state).to(device)

        for value_layer, actor_layer in zip(self.critic_net, self.actor_net):
          state_value = value_layer(state_value)
          action_logits = actor_layer(action_logits)

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

        entropy = (action_probs * action_logprobs).sum(1).mean()

        return action, entropy, action_logprobs, state_value


class ActorCriticNetContinuous(BaseNet, nn.Module):
  """
  Wrapper for the Actor-Critic neural networks (for a continuous action space)
  """

  def __init__(self):
    super().__init__()
    self.critic_net = None
    self.actor_net = None
    self.mu = None
    self.sigma = None
    self.clip_high = None
    self.clip_low = None


  def add_layers(self, nodes: list, env):
    if self.env_is_discrete(env):
        raise TypeError('Action space is discrete, not continuous - please use a discrete policy')

    self.clip_high = torch.tensor(env.action_space.high)
    self.clip_low = torch.tensor(env.action_space.low)

    self.actor_net = super().add_layers(nodes, env.observation_space.shape[0], 1)
    self.actor_net.pop(-1)
    self.critic_net = super().add_layers(nodes, env.observation_space.shape[0], 1)

    self.mu = nn.ModuleList([nn.Linear(nodes[-1], env.action_space.shape[0]),
                             nn.Tanh()])
    self.sigma = nn.ModuleList([nn.Linear(nodes[-1], env.action_space.shape[0]),
                                nn.Softplus()])


  def forward(self, state, device='mps'):
    state_value = torch.tensor(state).to(device)
    action_means = torch.tensor(state).to(device)

    for value_layer, base_layer in zip(self.critic_net[:-1], self.actor_net):
      state_value = value_layer(state_value)
      action_means = base_layer(action_means)

    state_value = self.critic_net[-1](state_value)
    action_stds = action_means.clone()

    for mu_layer, sigma_layer in zip(self.mu, self.sigma):
      action_means = mu_layer(action_means)
      action_stds = sigma_layer(action_stds)

    actions = torch.normal(action_means,
                          action_stds)
    actions = torch.clip(actions,
                         self.clip_low.to(device),
                         self.clip_high.to(device))

    action_logprobs = -(actions - action_means)**2 / (2 * (action_stds ** 2).clamp(min=1e-5))
    action_logprobs -= torch.log(torch.sqrt(2 * torch.pi * action_stds ** 2))

    entropy = torch.log(torch.sqrt(2 * torch.pi * torch.exp(torch.tensor(1)) * action_stds ** 2)).mean()

    return actions, entropy, action_logprobs, state_value


