from jax import Array

from core.agent import Model
from core.common import Params, InfoDict, Batch, filter_to_action
from update.loss import mc_mse_loss, td_mse_loss, expectile_loss

import jax.numpy as jnp
import jax
import optax

from typing import Tuple


def update_v(value: Model, batch: Batch, **kwargs) -> Tuple[Model, InfoDict]:
    """
    Function to update the Value network

    :param value: the Value network to be updated
    :param batch: a Batch object of samples
    :return: a tuple containing the new model parameters, plus metadata
    """

    loss_fn = {'mc_mse': mc_mse_loss,
               'td_mse': td_mse_loss,
               'expectile': expectile_loss}

    # Unpack the actions, states, and discounted rewards from the batch of samples
    states = batch.states
    len_actions = jnp.array([len(traj) for traj in batch.rewards]) - kwargs['interval_min']

    def value_loss_fn(value_params: Params) -> tuple[Array, dict[str, Array]]:
        # Generate V(s) for the sample states
        layer_outputs, v = value.apply({'params': value_params}, states)

        v = filter_to_action(v, len_actions)

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

    loss_fn = {'mc_mse': mc_mse_loss,
               'td_mse': td_mse_loss,
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

        # Calculate the loss for the critic networks using the target Q values with MSE
        critic_loss_1 = loss_fn[list(kwargs['critic_loss_fn'].keys())[0]](q1, batch, **kwargs).mean()
        critic_loss_2 = loss_fn[list(kwargs['critic_loss_fn'].keys())[0]](q2, batch, **kwargs).mean()
        critic_loss = critic_loss_1 + critic_loss_2

        # Return the loss value, plus metadata
        return critic_loss, {
            'critic_loss': critic_loss,
            'layer_outputs': layer_outputs,
        }

    # Calculate the updated model parameters using the loss function
    new_critic, info = critic.apply_gradient(critic_loss_fn)

    # Return the new model parameters, plus loss metadata
    return new_critic, info


def update_interval(interval: Model, batch: Batch, **kwargs) -> Tuple[Model, InfoDict]:
    """
    Function to update the Q network
    :param interval: the interval network to be updated
    :param batch: a Batch object of samples
    :return: a tuple containing the new model parameters, plus metadata
    """

    # Unpack the actions, states, and discounted rewards from the batch of samples
    states = batch.states
    len_actions = jnp.array([len(traj) for traj in batch.rewards]) - kwargs['interval_min']

    def interval_loss_fn(interval_params: Params) -> tuple[Array, dict[str, Array]]:
        # Generate Q values from each of the two critic networks
        layer_outputs, logits = interval.apply({'params': interval_params}, states)

        interval_loss = optax.softmax_cross_entropy_with_integer_labels(logits, len_actions).mean()

        # Return the loss value, plus metadata
        return interval_loss, {
            'interval_loss': interval_loss,
            'layer_outputs': layer_outputs,
        }

    # Calculate the updated model parameters using the loss function
    new_interval, info = interval.apply_gradient(interval_loss_fn)

    # Return the new model parameters, plus loss metadata
    return new_interval, info
