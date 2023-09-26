from core.common import Params
from typing import Sequence, Callable, Optional, Tuple, Any
import os

import jax.numpy as jnp
import numpy as np

import jax
from jax import grad, Array, jit
from flax import struct
import flax.linen as nn
from flax.training import orbax_utils
import optax
import orbax.checkpoint


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
    """
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[dict[int, Any], Array | Any]:
        """
        MLP forward pass.

        :param x: input to the MLP
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
            before_final_layer = i + 1 < len(self.hidden_dims) or self.activate_final

            x = jax.lax.select(
                before_final_layer,
                self.activations(x),
                x
            )

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
    output_dims: int = 1
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray, input_mean=0, input_std=1) -> tuple[dict[str, dict[int, Any]], Array]:
        """
        Value network forward pass.

        :param observations: input data for the forward pass
        :return: output of the value network
        """

        # Standardise the input data
        observations = (observations - input_mean) / input_std

        # Do a forward pass with the MLP
        layer_outputs, value = MLP((*self.hidden_dims, self.output_dims),
                                   activations=self.activations)(observations)

        # Return the output
        if value.shape[-1] == 1:
            return {'MLP_0': layer_outputs}, jnp.squeeze(value, -1)
        else:
            return {'MLP_0': layer_outputs}, value


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
    def __call__(self, observations: jnp.ndarray, input_mean=0, input_std=1) -> tuple[dict[str, dict[int, Any]], Any]:
        """
        Critic network forward pass.

        :param observations: input data for the forward pass
        :return: output of the critic network
        """

        """
        Because we only care about evaluating Q-values (and extracting policies) for known actions,
        we don't need to create Q-values for all possible actions. 
        
        Instead, we just treat the action as another input to the network.
        """
        # Standardise the input data
        observations = (observations - input_mean) / input_std

        # Forward pass
        layer_outputs, critic = MLP((*self.hidden_dims, self.action_dims),
                                    activations=self.activations)(observations)

        # Return the output of the MLP
        return {'MLP_0': layer_outputs}, critic


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
    def __call__(self, observations: jnp.ndarray, input_mean=0, input_std=1) -> tuple[dict[str, dict[int, Any]], tuple[Any, Any]]:
        """
        Double critic network forward pass.

        :param observations: input data for each critic's forward pass
        :return: output of each critic network
        """
        # Standardise the input data
        observations = (observations - input_mean) / input_std

        # Forward pass for each critic network MLP
        layer_outputs_q1, critic1 = MLP((*self.hidden_dims, self.action_dims),
                                        activations=self.activations)(observations)
        layer_outputs_q2, critic2 = MLP((*self.hidden_dims, self.action_dims),
                                        activations=self.activations)(observations)

        # Return both outputs
        return {'MLP_0': layer_outputs_q1,
                'MLP_1': layer_outputs_q2}, (critic1, critic2)


class ActorNet(nn.Module):
    """
    Actor network.

    Has attributes:
        - hidden_dims: the number of hidden units in each layer
        - activations: the activation function to use between layers
        - action_dims: the number of available actions
    """

    hidden_dims: Sequence[int]
    action_dims: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray, input_mean=0, input_std=1) -> tuple[dict[str, dict[int, Any]], Any]:
        """
        Actor network forward pass.

        :param observations: input data for the forward pass
        :return: output of the actor network
        """
        # Standardise the input data
        observations = (observations - input_mean) / input_std

        # Forward pass with the MLP
        layer_outputs, logits = MLP((*self.hidden_dims, self.action_dims))(observations)

        # Return the output
        return {'MLP_0': layer_outputs}, logits


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
    continual_learning: bool = False
    opt_state: Optional[optax.OptState] = None
    initializer = nn.initializers.he_normal()
    checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    ages: Sequence[jnp.ndarray] = None
    util: Sequence[jnp.ndarray] = None
    mean_outputs: Sequence[jnp.ndarray] = None
    bias_corrected_util: Sequence[jnp.ndarray] = None
    decay_rate: float = 0.99
    replacement_rate: float = 0.01
    maturity_threshold: int = 50
    input_mean: float = 0
    input_std: float = 1

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               optim: Optional[optax.GradientTransformation] = None,
               continual_learning: bool = False,
               ) -> 'Model':
        """
        Class method to create a new instance of the Model class (with some necessary pre-processing).

        :param model_def: the neural network architecture
        :param inputs: dummy input data to initialise the network
        :param optim: the optimiser
        :param continual_learning: whether to use continual learning
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
            empty_dict = {}
            for key, value in params_dict.items():
                empty_dict[key] = [jnp.zeros(value[f'Dense_{i}']['kernel'].shape[-1]) for i in
                                   range(len(value.keys()) - 1)]

            return empty_dict

        ages = iterate_through_layers(params)
        util = iterate_through_layers(params)
        mean_outputs = iterate_through_layers(params)
        bias_corrected_util = iterate_through_layers(params)

        # Return an instance of the class with the following attributes
        return cls(network=model_def,
                   params=params,
                   optim=optim,
                   continual_learning=continual_learning,
                   opt_state=opt_state,
                   ages=ages,
                   util=util,
                   mean_outputs=mean_outputs,
                   bias_corrected_util=bias_corrected_util)

    @jit
    def __call__(self, *args):
        """
        Forward pass through the neural network.

        :param args: input data to pass through the neural network
        :return: output of the neural network
        """

        return self.network.apply({'params': self.params}, *args,
                                  input_mean=self.input_mean,
                                  input_std=self.input_std)

    @jit
    def apply(self, *args, **kwargs):
        return self.network.apply(*args,
                                  input_mean=self.input_mean,
                                  input_std=self.input_std,
                                  **kwargs)

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
        """
        Work in progress - attempting to make continual learning a boolean option
        
        features_to_replace, num_features_to_replace = jax.lax.switch(self.continual_learning.astype(int),
                                                                      [lambda x, y: (None, None),
                                                                       self.choose_features,
                                                                       ], info['layer_outputs'], new_params)
        """

        features_to_replace, num_features_to_replace = self.choose_features(outputs=info['layer_outputs'],
                                                                            new_params=new_params)
        

        # Update the new parameters with re-initialised low utility nodes, as required
        new_params = self.gen_new_features(features_to_replace, new_params)
        
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

    def update_utility(self, layer_idx=0, outputs=None, params=None, key=None):
        """
        Update the utility of the nodes in the current layer.
        :param layer_idx: current layer number
        :param outputs: all output values (across all layers) from the neural network
        :param params: updated parameters of the neural network
        :return: None
        """

        # As per Equation 3/4/5 - we are using a running average, so we decay the current output values first
        self.mean_outputs[key][layer_idx] *= self.decay_rate
        # And then update with the new output values
        self.mean_outputs[key][layer_idx] += (1 - self.decay_rate) * outputs[layer_idx].mean(0)

        # We bias correct the output values
        bias_correction = 1 - self.decay_rate ** self.ages[key][layer_idx]
        bias_corrected_outputs = self.mean_outputs[key][layer_idx] / bias_correction

        # Then we calculate the utility of the nodes in the current layer, also using a running average
        self.util[key][layer_idx] *= self.decay_rate

        # Equation 6 - calculate the absolute difference between the current outputs and the running average
        new_util = jnp.absolute(outputs[layer_idx].mean(0) - bias_corrected_outputs)
        # Then multiply by the weights connecting the current outputs to the next layer
        new_util *= jnp.sum(jnp.absolute(params[f'Dense_{layer_idx + 1}']['kernel']), axis=1)
        # Then divide by the weights connecting the previous layer to the current outputs
        new_util /= jnp.sum(jnp.absolute(params[f'Dense_{layer_idx}']['kernel']), axis=0)

        # Update the utility with the new utility values
        self.util[key][layer_idx] += (1 - self.decay_rate) * new_util
        # And similarly update the bias-corrected utility
        self.bias_corrected_util[key][layer_idx] = self.util[key][layer_idx] / bias_correction

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
        features_to_replace = {}
        num_features_to_replace = {}

        for key, params in new_params.items():
            # Empty placeholders for features to replace and number of features to replace
            features_to_replace[key] = [jnp.empty(0, dtype=jnp.int32) for _ in range(n_hidden_layers)]
            num_features_to_replace[key] = [0 for _ in range(n_hidden_layers)]

            # Iterate through each layer
            for i in range(n_hidden_layers):
                # Update the age of the nodes in each layer
                self.ages[key][i] += 1

                # Update the utility of the nodes in each layer
                self.update_utility(layer_idx=i,
                                    outputs=outputs[key],
                                    params=params,
                                    key=key)

                # Find eligible features to replace in the current layer (assuming they are 'old' enough)
                eligible_features_bool = self.ages[key][i] > self.maturity_threshold

                # Calculate the number of features to replace in the current layer
                num_new_features_to_replace = self.replacement_rate * eligible_features_bool.sum()

                # If the number of features to replace in this layer is between 0-1, use this as a probability
                boolean_random_threshold = np.random.default_rng().uniform() <= num_new_features_to_replace
                num_new_features_to_replace = boolean_random_threshold * jnp.maximum(num_new_features_to_replace, 1)

                # Round to the lowest integer
                num_new_features_to_replace = num_new_features_to_replace.astype(int)

                # Rank the features in terms of utility
                ranked_features = jnp.argsort(self.bias_corrected_util[key][i])
                undo_ranking = jnp.argsort(ranked_features)

                # Combine this with the 'eligible node' boolean mask
                boolean_ranked_features = eligible_features_bool[ranked_features]

                # Select the top K eligible features
                top_k_features = jnp.arange(self.network.hidden_dims[i]) < num_new_features_to_replace
                new_features_to_replace = boolean_ranked_features * top_k_features

                # Revert back to the original ordering
                new_features_to_replace = new_features_to_replace[undo_ranking]

                # Set these as the features to replace
                features_to_replace[key][i] = new_features_to_replace
                num_features_to_replace[key][i] = num_new_features_to_replace

        return features_to_replace, num_features_to_replace

    def gen_new_features(self, features_to_replace, new_params):
        """
        Generate new features: Reset input and output weights for low utility features
        """

        def reset(array, index, replace=0):
            return jnp.where(index, replace, array)

        for key, params in new_params.items():

            # Iterate through each hidden layer
            for layer_idx in range(len(self.network.hidden_dims)):
                # Start by identifying the input and output layers
                current_layer = params[f'Dense_{layer_idx}']
                next_layer = params[f'Dense_{layer_idx + 1}']

                # Reset the input bias to 0
                current_layer['bias'] = reset(current_layer['bias'], features_to_replace[key][layer_idx])

                # Re-initialize the input weights randomly
                new_weights = default_initializer()(jax.random.PRNGKey(np.random.randint(int(1e9))),
                                                    shape=current_layer['kernel'].shape)

                current_layer['kernel'] = reset(current_layer['kernel'],
                                                features_to_replace[key][layer_idx].reshape(1, -1),
                                                new_weights)

                # Then set the output weights to zero (so the current performance is not directly affected)
                next_layer['kernel'] = reset(next_layer['kernel'], features_to_replace[key][layer_idx].reshape(-1, 1))

                # Reset the ages, mean outputs, and utility of the newly reset nodes to zero
                self.ages[key][layer_idx] = reset(self.ages[key][layer_idx], features_to_replace[key][layer_idx])
                self.util[key][layer_idx] = reset(self.util[key][layer_idx], features_to_replace[key][layer_idx])
                self.bias_corrected_util[key][layer_idx] = reset(self.bias_corrected_util[key][layer_idx],
                                                                 features_to_replace[key][layer_idx])
                self.mean_outputs[key][layer_idx] = reset(self.mean_outputs[key][layer_idx],
                                                          features_to_replace[key][layer_idx])

                new_params[key][f'Dense_{layer_idx}'] = current_layer
                new_params[key][f'Dense_{layer_idx + 1}'] = next_layer

        return new_params
