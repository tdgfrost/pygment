from agent import Model
from common import Params, InfoDict, Batch

import jax.numpy as jnp

from typing import Tuple


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

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Generate V(s) for the sample states
        v = value.apply({'params': value_params}, batch.states)

        # Calculate the loss for V using Q with expectile regression
        value_loss = loss(v - discounted_rewards, expectile).mean()

        # Return the loss value, plus metadata
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
        }

    # Calculate the updated model parameters using the loss function
    new_value, info = value.apply_gradient(value_loss_fn)

    # Return the new model parameters, plus loss metadata
    return new_value, info


def update_q(critic: Model, target_value: Model, batch: Batch,
             gamma: float) -> Tuple[Model, InfoDict]:

    # Calculate the optimal V(s') for the next states in the batch
    next_v = target_value(batch.next_states)

    # Use this to calculate the target Q(s,a) for each state in the batch under an optimal V(s')
    target_q = batch.rewards + gamma * ~batch.dones * next_v

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Generate Q values from each of the two critic networks
        q1, q2 = critic.apply({'params': critic_params}, batch.states,
                              batch.actions)

        # Calculate the loss for the critic networks using the target Q values with MSE
        critic_loss = ((q1 - target_q)**2).mean() + ((q2 - target_q)**2).mean()

        # Return the loss value, plus metadata
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    # Calculate the updated model parameters using the loss function
    new_critic, info = critic.apply_gradient(critic_loss_fn)

    # Return the new model parameters, plus loss metadata
    return new_critic, info
