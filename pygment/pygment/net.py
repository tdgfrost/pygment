from common import Params

import jax.numpy as jnp
import flax.linen as nn
from jax import grad
from flax.struct import dataclass, field

from typing import Sequence, Callable, Optional, Tuple, Any
import optax
import flax

import os


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x


class ValueNet(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1))(observations)
        return jnp.squeeze(critic, -1)


class CriticNet(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCriticNet(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = CriticNet(self.hidden_dims,
                            activations=self.activations)(observations, actions)
        critic2 = CriticNet(self.hidden_dims,
                            activations=self.activations)(observations, actions)
        return critic1, critic2


class ActorNet(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1))(observations)
        return jnp.squeeze(critic, -1)


@dataclass
class Model:
    network: nn.Module = field(pytree_node=False)  # Formerly apply_fn
    params: Params
    tx: Optional[optax.GradientTransformation] = field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None) -> 'Model':

        # Initialise the neural network
        variables = model_def.init(*inputs)

        # Extract the parameters (variables: dictionary has only one key, 'params')
        params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(network=model_def,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.network.apply({'params': self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.network.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple[Any, 'Model']:
        grad_fn = grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(params=new_params,
                            opt_state=new_opt_state), info

    """
    def save_model(self):
        pass

    def load_model(self,
                   path=None):
        if path is None:
            invalid_path = ''
            while True:
                path = input(invalid_path + 'Please enter model path, or press Q/q to exist: ')
                if path == 'Q' or path == 'q':
                    return
                if os.path.isfile(path):
                    break
                invalid_path = 'Invalid path - '

        self.path = os.path.dirname(path)
    """
    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'Model':
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)


"""
class BaseNet:
    '''
    Base neural network
    '''

    def __init__(self):
        super().__init__()
        self.has_net = False
        self.action_space = None
        self.observation_space = None

    @staticmethod
    def env_is_discrete(env):
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
    '''
    Wrapper around model to create both a main_net and a target_net
    '''

    def __init__(self):
        super().__init__()
        self.main_net = None
        self.target_net = None
        self.sync_tracker = 0

    def sync(self, alpha=0.01):
        if not self.has_net:
            return None
        if isinstance(alpha, int):
            if self.sync_tracker == alpha:
                self.target_net.load_state_dict(self.main_net.state_dict())
                self.sync_tracker = 0
            else:
                self.sync_tracker += 1

        else:
            for target_param, main_param in zip(self.target_net.parameters(), self.main_net.parameters()):
                target_param.data.copy_(alpha * main_param.data + (1.0 - alpha) * target_param.data)

    def add_layers(self, nodes: list, env):
        if not self.env_is_discrete(env):
            raise TypeError('Action space is continuous, not discrete - please use a continuous policy')

        self.main_net = super().add_base_layers(nodes, env.observation_space.shape[0], env.action_space.n)
        self.target_net = deepcopy(self.main_net)
        self.has_net = True

    def forward(self, state, target=False, device='cpu'):
        if not self.has_net:
            target = False
        net = self.target_net if target else self.main_net

        Q_s = torch.tensor(np.array(state)).to(device)

        for layer in net:
            Q_s = layer(Q_s)

        return Q_s


class PolicyGradientNet(BaseNet, nn.Module):
    '''
    Wrapper for a policy gradient-based neural network
    '''

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


class CriticNet(BaseNet, nn.Module):
    '''
    Wrapper for the Actor-Critic neural networks
    '''

    def __init__(self):
        super().__init__()
        self.critic_net = None
        self.has_net = False

    def add_layers(self, nodes: list, env):
        if not self.env_is_discrete(env):
            raise TypeError('Action space is continuous, not discrete - please use a continuous policy')

        self.critic_net = super().add_base_layers(nodes, env.observation_space.shape[0], 1)

    def forward(self, state, device='cpu'):
        state_value = torch.tensor(np.array(state)).to(device)

        for layer_idx in range(len(self.critic_net)):
            state_value = self.critic_net[layer_idx](state_value)

        return state_value


class ActorNet(BaseNet, nn.Module):
    '''
    Wrapper around model to create both a main_net and a target_net
    '''

    def __init__(self):
        super().__init__()
        self.actor_net = None
        self.sync_tracker = 0

    def add_layers(self, nodes: list, env):
        if not self.env_is_discrete(env):
            raise TypeError('Action space is continuous, not discrete - please use a continuous policy')

        self.actor_net = super().add_base_layers(nodes, env.observation_space.shape[0], env.action_space.n)
        self.has_net = True

    def forward(self, state, device='cpu'):
        logits = torch.tensor(np.array(state)).to(device)

        for layer in self.actor_net:
            logits = layer.to(device)(logits)

        return logits


class ActorCriticNet(BaseNet, nn.Module):
    '''
    Wrapper for the Actor-Critic neural networks
    '''

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
    '''
    Wrapper for the Actor-Critic neural networks (for a continuous action space)
    '''

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
"""
