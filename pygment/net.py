import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical


class BaseNet:
    """
    Base neural network
    """
    def __init__(self):
        self.activations = {'relu': nn.ReLU(),
                            'sigmoid': nn.Sigmoid(),
                            'leaky': nn.LeakyReLU()}

        self.output_activations = {'sigmoid': nn.Sigmoid(),
                                   'softmax': nn.Softmax(),
                                   'linear': None}


class DualNet(BaseNet):
    """
    Wrapper around model to create both a main_net and a target_net
    """

    def __init__(self, model=nn.Sequential()):
        super().__init__()
        self.main_net = model
        self.target_net = None
        self.action_space = None
        self.observation_space = None


    def sync(self, tau):
        for target_param, main_param in zip(self.target_net.parameters(), self.main_net.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

        #self.target_net.load_state_dict(self.main_net.state_dict())


class ActorCriticNet(BaseNet, nn.Module):
    """
    Wrapper for the Actor-Critic neural networks
    """

    def __init__(self):
        super().__init__()

        self.input_layer = nn.Linear(8, 128)

        self.action_layer = nn.Linear(128, 4)
        self.value_layer = nn.Linear(128, 1)


    def forward(self, state):
        state = torch.tensor(state)
        state = F.relu(self.input_layer(state))

        state_value = self.value_layer(state)

        action_probs = F.softmax(self.action_layer(state))
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()

        return action.item()


    def calc
