import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical


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


    def add_layers(self, nodes: list, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

        input_layers = nn.ModuleList([])

        for layer in range(len(nodes)):
            if layer == 0:
                input_layers.append(nn.Linear(self.observation_space, nodes[layer]))
            else:
                input_layers.append(nn.Linear(nodes[layer - 1], nodes[layer]))

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


    def add_layers(self, nodes: list, observation_space, action_space):
        self.main_net = super().add_layers(nodes, observation_space, action_space)


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


    def add_layers(self, nodes: list, observation_space, action_space):
        self.net = super().add_layers(nodes, observation_space, action_space)


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
        self.net = None
        self.action_layer = None
        self.value_layer = None


