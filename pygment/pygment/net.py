from common import Params

import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from jax import grad, Array
from flax import struct

from typing import Sequence, Callable, Optional, Tuple, Any, Dict
import optax
import orbax.checkpoint
from flax.training import orbax_utils
import jax


def default_initializer():
    """
    Default initialiser for the neural network weights.
    """

    return nn.initializers.he_normal()


class MLP(nn.Module):
    """
    Multi-layer perceptron.
    Has attributes:
        - hidden_dims: the number of hidden units in each layer
        - activations: the activation function to use between layers
        - activate_final: whether to apply the activation function to the final layer
        - dropout_rate: the dropout rate to apply between layers
    """
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> tuple[dict[int, Any], Array | Any]:
        """
        MLP forward pass.

        :param x: input to the MLP
        :param training: whether to keep dropout random (during training), or consistent (during evaluation)
        :return: output of the MLP
        """

        layer_outputs = {}

        # Iterate through each layer
        for i, size in enumerate(self.hidden_dims):

            # Apply a dense layer
            """
            If using LSTM / RNN, consider switching initialiser to orthogonal.
            He normal is good for ReLU, but not necessarily for other activation functions.
            Glorot normal is good for tanh / sigmoid.
            """
            x = nn.Dense(size, kernel_init=default_initializer())(x)

            # Apply the activation function
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)

                # Apply dropout (deterministically during evaluation)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

            layer_outputs[i] = x.copy()

        return layer_outputs, x


class ValueNet(nn.Module):
    """
    Value network.

    Has attributes:
        - hidden_dims: the number of hidden units in each layer
        - activations: the activation function to use between layers
    """

    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> tuple[dict[int, Any], Array]:
        """
        Value network forward pass.

        :param observations: input data for the forward pass
        :return: output of the value network
        """

        # Do a forward pass with the MLP
        layer_outputs, value = MLP((*self.hidden_dims, 1),
                                   activations=self.activations)(observations)

        # Return the output
        return layer_outputs, jnp.squeeze(value, -1)


class CriticNet(nn.Module):
    """
    Critic network.

    Has attributes:
        - hidden_dims: the number of hidden units in each layer
        - action_dims: the number of available actions
        - activations: the activation function to use between layers
    """

    hidden_dims: Sequence[int]
    action_dims: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> tuple[dict[int, Any], Any]:
        """
        Critic network forward pass.

        :param observations: input data for the forward pass
        :param actions: chosen actions to evaluate the Q-value for
        :return: output of the critic network
        """

        """
        Because we only care about evaluating Q-values (and extracting policies) for known actions,
        we don't need to create Q-values for all possible actions. 
        
        Instead, we just treat the action as another input to the network.
        """
        layer_outputs, critic = MLP((*self.hidden_dims, self.action_dims),
                                    activations=self.activations)(observations)

        # Return the output of the MLP
        return layer_outputs, critic


class DoubleCriticNet(nn.Module):
    """
    Double critic network.

    Has attributes:
        - hidden_dims: the number of hidden units in each layer
        - action_dims: the number of available actions
        - activations: the activation function to use between layers
    """

    hidden_dims: Sequence[int]
    action_dims: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> tuple[tuple[dict[int, Any], dict[int, Any]], tuple[Any, Any]]:
        """
        Double critic network forward pass.

        :param observations: input data for each critic's forward pass
        :param actions: actions to select the Q-value for
        :return: output of each critic network
        """

        # Forward pass for each critic network MLP
        layer_outputs_q1, critic1 = CriticNet(self.hidden_dims, self.action_dims,
                                              activations=self.activations)(observations)
        layer_outputs_q2, critic2 = CriticNet(self.hidden_dims, self.action_dims,
                                              activations=self.activations)(observations)

        # Return both outputs
        return (layer_outputs_q1, layer_outputs_q2), (critic1, critic2)


class ActorNet(nn.Module):
    """
    Actor network.

    Has attributes:
        - hidden_dims: the number of hidden units in each layer
        - activations: the activation function to use between layers
        - dropout_rate: the dropout rate to apply between layers
        - action_dims: the number of available actions
    """

    hidden_dims: Sequence[int]
    action_dims: int
    dropout_rate: Optional[float] = None
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray, training: bool = False) -> tuple[dict[int, Any], Any]:
        """
        Actor network forward pass.

        :param observations: input data for the forward pass
        :return: output of the actor network
        """

        # Forward pass with the MLP
        layer_outputs, logits = MLP((*self.hidden_dims, self.action_dims),
                                    dropout_rate=self.dropout_rate)(observations,
                                                                    training=training)

        # Return the output
        return layer_outputs, logits


@struct.dataclass
class Model:
    """
    Model class, which contains the neural network, parameters, and optimiser state.

    Has attributes:
        - network: the neural network architecture
        - params: the parameters of the neural network
        - optim: the optimiser
        - opt_state: the optimiser state
    """

    network: nn.Module = struct.field(pytree_node=False)
    params: Params
    optim: Optional[optax.GradientTransformation] = struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None
    initializer = nn.initializers.he_normal()
    checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    ages: Sequence[jnp.ndarray] = None
    util: Sequence[jnp.ndarray] = None
    mean_outputs: Sequence[jnp.ndarray] = None
    bias_corrected_util: Sequence[jnp.ndarray] = None
    decay_rate: float = 0.99
    replacement_rate: float = 0.01
    maturity_threshold: int = 20

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               optim: Optional[optax.GradientTransformation] = None
               ) -> 'Model':
        """
        Class method to create a new instance of the Model class (with some necessary pre-processing).

        :param model_def: the neural network architecture
        :param inputs: dummy input data to initialise the network
        :param optim: the optimiser
        :return: an instance of the Model class
        """

        # Initialise the neural network
        variables = model_def.init(*inputs)

        # Extract the parameters (variables: dictionary has only one key, 'params')
        params = variables.pop('params')

        # Initialise the optimiser state if optimiser is present
        if optim is not None:
            opt_state = optim.init(params)
        else:
            opt_state = None

        # Define
        def iterate_through_layers(params_dict):
            empty_list = []
            for key, value in params_dict.items():
                if 'Dense_0' in value.keys():
                    return [jnp.zeros(value[f'Dense_{i}']['kernel'].shape[-1]) for i in range(len(value.keys()) - 1)]

                else:
                    empty_list.append(iterate_through_layers(params_dict[key]))

            return empty_list

        ages = iterate_through_layers(params)
        util = ages.copy()
        mean_outputs = ages.copy()
        bias_corrected_util = ages.copy()

        # Return an instance of the class with the following attributes
        return cls(network=model_def,
                   params=params,
                   optim=optim,
                   opt_state=opt_state,
                   ages=ages,
                   util=util,
                   mean_outputs=mean_outputs,
                   bias_corrected_util=bias_corrected_util)

    def __call__(self, *args):
        """
        Forward pass through the neural network.

        :param args: input data to pass through the neural network
        :return: output of the neural network
        """

        return self.network.apply({'params': self.params}, *args)

    def apply(self, *args, **kwargs):
        return self.network.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple['Model', Any]:
        """
        Calculate the gradient of the loss function with respect to the parameters of the neural network.
        Returns the updated parameters and optimiser state.

        :param loss_fn: loss function used to calculate the gradient
        :return: updated parameters and optimiser state
        """

        # Convert the loss function to a gradient function
        grad_fn = grad(loss_fn, has_aux=True)

        # Calculate the gradients and relevant metadata
        grads, info = grad_fn(self.params)

        # Calculate the parameter updates, as well as the new optimiser state
        updates, new_opt_state = self.optim.update(grads, self.opt_state,
                                                   self.params)

        # Calculate the new parameters
        new_params = optax.apply_updates(self.params, updates)

        # Identify the lowest utility nodes to be replaced - as per "Loss of Plasticity in Deep Continual Learning"
        #features_to_replace, num_features_to_replace = self.choose_features(outputs=info['layer_outputs'],
                                                                            #new_params=new_params)

        # Update the new parameters with re-initialised low utility nodes, as required
        #new_params = self.gen_new_features(features_to_replace, num_features_to_replace, new_params)

        # Returns a COPY with the new parameters and optimiser state, as well as the metadata
        return self.replace(params=new_params,
                            opt_state=new_opt_state), info

    def save(self, save_path: str):
        self.checkpointer.save(save_path, self.params,
                               save_args=orbax_utils.save_args_from_target(self.params),
                               force=True)

    def load(self, load_path: str) -> 'Model':
        # Load the saved parameters
        new_params = self.checkpointer.restore(load_path)

        # Returns a COPY with the updated parameters
        return self.replace(params=new_params)

    def update_utility(self, layer_idx=0, outputs=None, new_params=None):
        """
        Update the utility of the nodes in the current layer.
        :param layer_idx: current layer number
        :param outputs: all output values (across all layers) from the neural network
        :param new_params: updated parameters of the neural network
        :return: None
        """

        # As per Equation 3/4/5 - we are using a running average, so we decay the current output values first
        self.mean_outputs[layer_idx] *= self.decay_rate
        # And then update with the new output values
        self.mean_outputs[layer_idx] += (1 - self.decay_rate) * outputs[layer_idx].mean(0)

        # We bias correct the output values
        bias_correction = 1 - self.decay_rate ** self.ages[layer_idx]
        bias_corrected_outputs = self.mean_outputs[layer_idx] / bias_correction

        # Then we calculate the utility of the nodes in the current layer, also using a running average
        self.util[layer_idx] *= self.decay_rate

        # Equation 6 - calculate the absolute difference between the current outputs and the running average
        new_util = jnp.absolute(outputs[layer_idx].mean(0) - bias_corrected_outputs)
        # Then multiply by the weights connecting the current outputs to the next layer
        new_util *= jnp.sum(jnp.absolute(new_params['MLP_0'][f'Dense_{layer_idx + 1}']['kernel']), axis=1)
        # Then divide by the weights connecting the previous layer to the current outputs
        new_util /= jnp.sum(jnp.absolute(new_params['MLP_0'][f'Dense_{layer_idx}']['kernel']), axis=0)

        # Update the utility with the new utility values
        self.util[layer_idx] += (1 - self.decay_rate) * new_util
        # And similarly update the bias-corrected utility
        self.bias_corrected_util[layer_idx] = self.util[layer_idx] / bias_correction

    def choose_features(self, outputs, new_params=None):
        """
        Update the utility values of all the nodes, and choose the lowest utility nodes to re-initialise.

        :param outputs: outputs from each layer of the neural network
        :param new_params: updated parameters for the neural network
        :return: feature indexes (for each layer) to reset, and the number of features being reset
        """
        # Define number of layers
        n_hidden_layers = len(self.network.hidden_dims)
        # Empty placeholders for features to replace and number of features to replace
        features_to_replace = [jnp.empty(0, dtype=jnp.int32) for _ in range(n_hidden_layers)]
        num_features_to_replace = [0 for _ in range(n_hidden_layers)]

        # Otherwise, iterate through each layer
        for i in range(n_hidden_layers):
            # Update the age of the nodes in each layer
            self.ages[i] += 1

            # Update the utility of the nodes in each layer
            self.update_utility(layer_idx=i, outputs=outputs, new_params=new_params)

            # Find eligible features to replace in the current layer (assuming they are 'old' enough)
            eligible_features_bool = self.ages[i] > self.maturity_threshold

            # If no features old enough to be replaced, continue to the next layer
            # if eligible_feature_indices.shape[0] == 0:
                # continue

            # Calculate the number of features to replace in the current layer
            num_new_features_to_replace = self.replacement_rate * eligible_features_bool.sum()

            # If the number of features to replace in this layer is between 0-1, use this as a probability
            boolean_random_threshold = np.random.default_rng().uniform() <= num_new_features_to_replace
            num_new_features_to_replace = boolean_random_threshold * jnp.maximum(num_new_features_to_replace, 1)

            # Otherwise, just round to the lowest integer
            num_new_features_to_replace = num_new_features_to_replace.astype(int)

            # If no features are being replaced, continue to the next layer
            # if num_new_features_to_replace == 0:
                # continue

            # Otherwise, find the features to replace by choosing the K lowest utility nodes
            ranked_features = jnp.argsort(self.bias_corrected_util[i])
            undo_ranking = jnp.argsort(ranked_features)
            boolean_ranked_features = eligible_features_bool[ranked_features]

            top_k_features = jnp.arange(self.network.hidden_dims[i]) < num_new_features_to_replace

            new_features_to_replace = boolean_ranked_features * top_k_features

            new_features_to_replace = new_features_to_replace[undo_ranking]

            # new_features_to_replace = jax.lax.top_k(-self.bias_corrected_util[i][eligible_feature_bool],
                                                     # num_new_features_to_replace)[1]

            # Update the feature indices to those specific features
            # new_features_to_replace = eligible_feature_indices[new_features_to_replace]

            # Reset the utility and mean outputs of these nodes to zero
            # self.util[i] = self.util[i].at[new_features_to_replace].set(0)
            # self.mean_outputs[i] = self.mean_outputs[i].at[new_features_to_replace].set(0)

            features_to_replace[i] = new_features_to_replace
            num_features_to_replace[i] = num_new_features_to_replace

        return features_to_replace, num_features_to_replace

    def gen_new_features(self, features_to_replace, num_features_to_replace, new_params):
        """
        Generate new features: Reset input and output weights for low utility features
        """
        def reset(array, index, replace=0):
            return jnp.where(index, replace, array)

        # Iterate through each hidden layer
        for layer_idx in range(len(self.network.hidden_dims)):

            # If no features being replaced in this layer, continue
            # if num_features_to_replace[layer_idx] == 0:
                # continue

            # Otherwise, reset the input and output weights for the low utility features

            # Start by identifying the input and output layers
            current_layer = new_params['MLP_0'][f'Dense_{layer_idx}']
            next_layer = new_params['MLP_0'][f'Dense_{layer_idx + 1}']

            # Reset the input bias to 0
            current_layer['bias'] = reset(current_layer['bias'], features_to_replace[layer_idx])

            # Re-initialize the input weights randomly
            new_weights = default_initializer()(jax.random.PRNGKey(0),
                                                shape=current_layer['kernel'].shape)

            current_layer['kernel'] = reset(current_layer['kernel'], features_to_replace[layer_idx].reshape(1, -1),
                                            new_weights)

            # Then set the output weights to zero (so the current performance is not directly affected)
            next_layer['kernel'] = reset(next_layer['kernel'], features_to_replace[layer_idx].reshape(-1, 1))

            # Reset the ages, mean outputs, and utility of the newly reset nodes to zero
            self.ages[layer_idx] = reset(self.ages[layer_idx], features_to_replace[layer_idx])
            self.util[layer_idx] = reset(self.util[layer_idx], features_to_replace[layer_idx])
            self.bias_corrected_util[layer_idx] = reset(self.bias_corrected_util[layer_idx],
                                                        features_to_replace[layer_idx])
            self.mean_outputs[layer_idx] = reset(self.mean_outputs[layer_idx], features_to_replace[layer_idx])

            new_params['MLP_0'][f'Dense_{layer_idx}'] = current_layer
            new_params['MLP_0'][f'Dense_{layer_idx + 1}'] = next_layer

        return new_params


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