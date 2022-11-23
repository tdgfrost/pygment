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


class DualNet(BaseNet):
    """
    Wrapper around model to create both a main_net and a target_net
    """

    def __init__(self, model=nn.Sequential()):
        super().__init__()
        self.main_net = model
        self.target_net = None


    def sync(self, tau):
        for target_param, main_param in zip(self.target_net.parameters(), self.main_net.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

        # At some point, add a function to allow soft vs hard updates between
        # main net and target net. Below is the hard update code.
        #self.target_net.load_state_dict(self.main_net.state_dict())


class PolicyGradientNet(BaseNet, nn.Module):
    """
    Wrapper for a policy gradient-based neural network
    """

    def __init__(self):
        super().__init__()
        self.input_layers = nn.ModuleList([])


    def add_layers(self, layers: int, nodes: list, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

        for layer in range(layers):
            if layer == 0:
                self.input_layers.append(nn.Linear(self.observation_space, nodes[layer]))
            else:
                self.input_layers.append(nn.Linear(nodes[layer - 1], nodes[layer]))

        self.input_layers.append(nn.Linear(nodes[-1], self.action_space))


    def forward(self, state, device='mps'):
        state = torch.tensor(state).to(device)

        for layer in self.input_layers[:-1]:
            state = F.relu(layer(state))

        action_logits = self.input_layers[-1](state)
        action_probs = F.softmax(action_logits, dim=0)
        action_logprobs = F.log_softmax(action_logits, dim=0)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()

        return action.item(), action_probs, action_logprobs


class ActorCriticNet(BaseNet, nn.Module):
    """
    Wrapper for the Actor-Critic neural networks
    """

    def __init__(self):
        super().__init__()
        self.input_layers = nn.ModuleList([])
        self.action_layer = None
        self.value_layer = None


    def add_layers(self, layers: int, nodes: list, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

        for layer in range(layers):
            if layer == 0:
                self.input_layers.append(nn.Linear(self.observation_space, nodes[layer]))
            else:
                self.input_layers.append(nn.Linear(nodes[layer-1], nodes[layer]))

        self.action_layer = nn.Linear(nodes[-1], self.action_space)
        self.value_layer = nn.Linear(nodes[-1], 1)


    def forward(self, state):
        state = torch.tensor(state)

        for layer in self.input_layers:
            state = F.relu(layer(state))

        state_value = self.value_layer(state)

        action_probs = F.softmax(self.action_layer(state), dim=0)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()

        return action.item(), state_value

