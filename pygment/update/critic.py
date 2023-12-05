from jax import Array
import jax.numpy as jnp

from core.agent import Model
from core.common import Params, InfoDict, Batch, filter_to_action
from update.loss import mse_loss, expectile_loss

from typing import Tuple


def update_v(value: Model, batch: Batch, **kwargs) -> Tuple[Model, InfoDict]:
    """
    Function to update the Value network

    :param value: the Value network to be updated
    :param batch: a Batch object of samples
    :return: a tuple containing the new model parameters, plus metadata
    """

    loss_fn = {'mse': mse_loss,
               'expectile': expectile_loss}

    # Unpack the actions, states, and discounted rewards from the batch of samples
    states = batch.states

    def value_loss_fn(value_params: Params) -> tuple[Array, dict[str, Array]]:
        # Generate V(s) for the sample states
        layer_outputs, v = value.apply({'params': value_params}, states)

        if batch.len_actions is not None and len(v.shape) > 1:
            v = filter_to_action(v, batch.len_actions)

        # Calculate the loss for V using Q with expectile regression
        all_value_loss = loss_fn[list(kwargs['value_loss_fn'].keys())[0]](v, batch, **kwargs)
        value_loss = all_value_loss.mean()

        # Return the loss value, plus metadata
        return value_loss, {
            'value_loss': value_loss,
            'all_value_loss': all_value_loss,
            'layer_outputs': layer_outputs,
        }

    # Calculate the updated model parameters using the loss function
    new_value, info = value.apply_gradient(value_loss_fn)

    # Return the new model parameters, plus loss metadata
    return new_value, info


def update_q(critic: Model, batch: Batch, **kwargs) -> Tuple[Model, InfoDict]:
    """
    Function to update the Q network
    :param critic: the critic network to be updated
    :param batch: a Batch object of samples
    :return: a tuple containing the new model parameters, plus metadata
    """

    # Unpack the actions, states, and discounted rewards from the batch of samples
    actions = batch.actions
    states = batch.states

    loss_fn = {'mse': mse_loss,
               'expectile': expectile_loss}

    def critic_loss_fn(critic_params: Params) -> tuple[Array, dict[str, Array]]:
        # Generate Q values from each of the two critic networks
        layer_outputs, (q1, q2) = critic.apply({'params': critic_params}, states)

        # Select the sampled actions
        """
        This is just until jnp.take_along_axis works on metal again
        
        q1 = jnp.squeeze(jnp.take_along_axis(q1, actions, axis=-1), -1)
        q2 = jnp.squeeze(jnp.take_along_axis(q2, actions, axis=-1), -1)
        """

        q1 = filter_to_action(q1, actions)
        q2 = filter_to_action(q2, actions)
        q_min_bool = q1 < q2

        # Calculate the loss for the critic networks using the target Q values with MSE
        all_critic_loss_1 = loss_fn[list(kwargs['critic_loss_fn'].keys())[0]](q1, batch, **kwargs)
        all_critic_loss_2 = loss_fn[list(kwargs['critic_loss_fn'].keys())[0]](q2, batch, **kwargs)
        critic_loss = all_critic_loss_1.mean() + all_critic_loss_2.mean()

        all_critic_loss = jnp.where(q_min_bool, all_critic_loss_1, all_critic_loss_2)

        # Return the loss value, plus metadata
        return critic_loss, {
            'critic_loss': critic_loss,
            'all_critic_loss': all_critic_loss,
            'layer_outputs': layer_outputs,
        }

    # Calculate the updated model parameters using the loss function
    new_critic, info = critic.apply_gradient(critic_loss_fn)

    # Return the new model parameters, plus loss metadata
    return new_critic, info


def update_uncertainty(uncertainty: Model, batch: Batch, target_loss) -> Tuple[Model, InfoDict]:
    """
    Function to update the Value network

    :param uncertainty: the Uncertainty network to be updated
    :param batch: a Batch object of samples
    :param target_loss: the target loss_value
    :return: a tuple containing the new model parameters, plus metadata
    """

    # Unpack the actions, states, and discounted rewards from the batch of samples
    states = batch.states

    def uncertainty_loss_fn(uncertainty_params: Params) -> tuple[Array, dict[str, Array]]:
        # Generate U(s) for the sample states
        layer_outputs, u = uncertainty.apply({'params': uncertainty_params}, states)

        if batch.actions is not None and len(u.shape) > 1:
            u = filter_to_action(u, batch.actions)

        # Regress the loss value (using MSE)
        uncertainty_loss = ((u - target_loss)**2).mean()

        # Return the loss value, plus metadata
        return uncertainty_loss, {
            'uncertainty_loss': uncertainty_loss,
            'layer_outputs': layer_outputs,
        }

    # Calculate the updated model parameters using the loss function
    new_uncertainty, info = uncertainty.apply_gradient(uncertainty_loss_fn)

    # Return the new model parameters, plus loss metadata
    return new_uncertainty, info