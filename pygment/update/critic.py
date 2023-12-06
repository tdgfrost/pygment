import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey

from core.agent import Model
from core.common import Params, InfoDict, Batch, filter_to_action
from update.loss import mse_loss, expectile_loss, gaussian_mse_loss, gaussian_expectile_loss

from typing import Tuple


def update_v(rng: PRNGKey, value: Model, batch: Batch, **kwargs) -> Tuple[Model, InfoDict]:
    """
    Function to update the Value network

    :param rng: the random number generator
    :param value: the Value network to be updated
    :param batch: a Batch object of samples
    :return: a tuple containing the new model parameters, plus metadata
    """

    loss_fn = {'mse': mse_loss,
               'expectile': expectile_loss,
               'gaussian_mse': gaussian_mse_loss,
               'gaussian_expectile': gaussian_expectile_loss}

    # Unpack the actions, states, and discounted rewards from the batch of samples
    states = batch.states

    def value_loss_fn(value_params: Params) -> tuple[Array, dict[str, Array]]:
        # Generate V(s) for the sample states
        layer_outputs, v = value.apply({'params': value_params}, states, rngs={'dropout': rng})

        if 'gaussian' in list(kwargs['value_loss_fn'].keys())[0]:
            v_s = [jnp.expand_dims(v, 0)]
            key, new_rng = jax.random.split(rng)
            for _ in range(9):
                layer_outputs, v = value.apply({'params': value_params}, states, rngs={'dropout': new_rng})
                v_s.append(jnp.expand_dims(v, 0))
                key, new_rng = jax.random.split(key)

            v_s = jnp.concatenate(v_s, axis=0)
            v_mu = jnp.mean(v_s, axis=0)
            v_sigma = jnp.std(v_s, axis=0)

            # Calculate the loss for V using Q with expectile regression
            value_loss = loss_fn[list(kwargs['value_loss_fn'].keys())[0]]((v_mu, v_sigma), batch, **kwargs).mean()
        else:
            # Calculate the loss for V using Q with expectile regression
            value_loss = loss_fn[list(kwargs['value_loss_fn'].keys())[0]](v, batch, **kwargs).mean()

        # Return the loss value, plus metadata
        return value_loss, {
            'value_loss': value_loss,
            'layer_outputs': layer_outputs,
        }

    # Calculate the updated model parameters using the loss function
    new_value, info = value.apply_gradient(value_loss_fn)

    # Return the new model parameters, plus loss metadata
    return new_value, info


def update_q(rng: PRNGKey, critic: Model, batch: Batch, **kwargs) -> Tuple[Model, InfoDict]:
    """
    Function to update the Q network

    :param rng: the random number generator
    :param critic: the critic network to be updated
    :param batch: a Batch object of samples
    :return: a tuple containing the new model parameters, plus metadata
    """

    # Unpack the actions, states, and discounted rewards from the batch of samples
    actions = batch.actions
    states = batch.states

    loss_fn = {'mse': mse_loss,
               'expectile': expectile_loss,
               'gaussian_mse': gaussian_mse_loss,
               'gaussian_expectile': gaussian_expectile_loss}

    def critic_loss_fn(critic_params: Params) -> tuple[Array, dict[str, Array]]:
        # Generate Q values from each of the two critic networks
        layer_outputs, q = critic.apply({'params': critic_params}, states, rngs={'dropout': rng})

        # Select the sampled actions
        """
        This is just until jnp.take_along_axis works on metal again
        
        q1 = jnp.squeeze(jnp.take_along_axis(q1, actions, axis=-1), -1)
        q2 = jnp.squeeze(jnp.take_along_axis(q2, actions, axis=-1), -1)
        """

        q = filter_to_action(q, actions)

        if 'gaussian' in list(kwargs['critic_loss_fn'].keys())[0]:
            q_s = [jnp.expand_dims(q, 0)]
            key, new_rng = jax.random.split(rng)
            for _ in range(9):
                layer_outputs, q = critic.apply({'params': critic_params}, states, rngs={'dropout': new_rng})
                q = filter_to_action(q, actions)

                q_s.append(jnp.expand_dims(q, 0))
                key, new_rng = jax.random.split(key)

            q_s = jnp.concatenate(q_s, axis=0)
            q_mu = jnp.mean(q_s, axis=0)
            q_sigma = jnp.std(q_s, axis=0)

            # Calculate the loss for V using Q with expectile regression
            critic_loss = loss_fn[list(kwargs['critic_loss_fn'].keys())[0]]((q_mu, q_sigma), batch, **kwargs).mean()
        else:
            # Calculate the loss for the critic networks using the target Q values with MSE
            critic_loss = loss_fn[list(kwargs['critic_loss_fn'].keys())[0]](q, batch, **kwargs).mean()

        # Return the loss value, plus metadata
        return critic_loss, {
            'critic_loss': critic_loss,
            'layer_outputs': layer_outputs,
        }

    # Calculate the updated model parameters using the loss function
    new_critic, info = critic.apply_gradient(critic_loss_fn)

    # Return the new model parameters, plus loss metadata
    return new_critic, info

