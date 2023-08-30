from jax import Array

from agent import Model
from common import Params, InfoDict, Batch

import jax.numpy as jnp
import jax

from typing import Tuple, Dict


def loss(diff, expectile=0.8):
    """
    Calculates the expectile of the Cauchy loss function for the residuals.
    Uses the formula L[ c^2/2 * log(loss^2 / c^2 + 1)]
    In this case, c is set to sqrt(2), so the formula simplifies to L[log(loss^2/2 + 1)]

    :param diff: the error term
    :param expectile: the expectile value
    :return: the loss term
    """

    weight = jnp.where(diff > 0, (1 - expectile), expectile)
    return weight * jnp.log(diff ** 2 / 2 + 1)


def update_v(value: Model, batch: Batch, expectile: float) -> Tuple[Model, InfoDict]:
    """
    Function to update the Value network

    :param value: the Value network to be updated
    :param batch: a Batch object of samples
    :param expectile: the value of tau for expectile regression
    :return: a tuple containing the new model parameters, plus metadata
    """
    # Unpack the actions and discounted rewards from the batch of samples
    discounted_rewards = batch.discounted_rewards

    def value_loss_fn(value_params: Params) -> tuple[Array, dict[str, Array]]:
        # Generate V(s) for the sample states
        v = value.apply({'params': value_params}, batch.states)

        # Calculate the loss for V using Q with expectile regression
        value_loss = loss(v - discounted_rewards, expectile).mean()

        # Return the loss value, plus metadata
        return value_loss, {
            'value_loss': value_loss,
        }

    # Calculate the updated model parameters using the loss function
    new_value, info = value.apply_gradient(value_loss_fn)

    # Return the new model parameters, plus loss metadata
    return new_value, info


def update_q(critic: Model, target_value: Model, batch: Batch,
             gamma: float) -> Tuple[Model, InfoDict]:

    # Unpack the states and actions
    states = batch.states
    actions = batch.actions

    # Calculate the optimal V(s') for the next states in the batch
    next_v = target_value(batch.next_states)

    # Use this to calculate the target Q(s,a) for each state in the batch under an optimal V(s')
    target_q = batch.rewards + gamma * (~batch.dones).astype(jnp.float32) * next_v

    def critic_loss_fn(critic_params: Params) -> tuple[Array, dict[str, Array]]:
        # Generate Q values from each of the two critic networks
        q1, q2 = critic.apply({'params': critic_params}, states)

        # Select the sampled actions
        """
        This is just until jnp.take_along_axis works on metal again
        
        q1 = jnp.squeeze(jnp.take_along_axis(q1, actions, axis=-1), -1)
        q2 = jnp.squeeze(jnp.take_along_axis(q2, actions, axis=-1), -1)
        """

        q1 = jax.lax.gather(q1,
                            jnp.concatenate((jnp.arange(len(actions)).reshape(-1, 1), actions), axis=1),
                            jax.lax.GatherDimensionNumbers(
                                offset_dims=tuple([]),
                                collapsed_slice_dims=tuple([0, 1]),
                                start_index_map=tuple([0, 1])), tuple([1, 1]))
        q2 = jax.lax.gather(q2,
                            jnp.concatenate((jnp.arange(len(actions)).reshape(-1, 1), actions), axis=1),
                            jax.lax.GatherDimensionNumbers(
                                offset_dims=tuple([]),
                                collapsed_slice_dims=tuple([0, 1]),
                                start_index_map=tuple([0, 1])), tuple([1, 1]))

        # Calculate the loss for the critic networks using the target Q values with MSE
        critic_loss = ((q1 - target_q)**2).mean() + ((q2 - target_q)**2).mean()

        # Return the loss value, plus metadata
        return critic_loss, {
            'critic_loss': critic_loss
        }

    # Calculate the updated model parameters using the loss function
    new_critic, info = critic.apply_gradient(critic_loss_fn)

    # Return the new model parameters, plus loss metadata
    return new_critic, info
