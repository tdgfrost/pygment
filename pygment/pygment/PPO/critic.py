from jax import Array

from agent import Model
from common import Params, InfoDict, Batch

import numpy as np
import jax.numpy as jnp
import jax

from typing import Tuple, Dict


def expectile_loss(diff, expectile=0.8):
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


def mse_loss(diff):
    """
    Calculates the mean squared error loss function for the residuals.

    :param diff: the error term
    :return: the loss term
    """

    return (diff ** 2).mean()


def update_v(value: Model, batch: Batch) -> Tuple[Model, InfoDict]:
    """
    Function to update the Value network

    :param value: the Value network to be updated
    :param batch: a Batch object of samples
    :param gamma: the discount factor
    :return: a tuple containing the new model parameters, plus metadata
    """
    # Get the discounted future reward

    target_v = batch.discounted_rewards
    """
    _, next_v = value(batch.next_states)

    max_seq_len = max([len(seq) for seq in batch.rewards])
    seq_rewards = jnp.array([[seq[i] if i < len(seq) else 0 for i in range(max_seq_len)] for seq in batch.rewards])

    discounted_rewards = jax.lax.scan(lambda agg, reward: (agg * gamma + reward, agg * gamma + reward),
                                      jnp.zeros(shape=seq_rewards.shape[0]), seq_rewards.transpose(),
                                      reverse=True)[1].transpose()[:, 0]

    gammas = jnp.ones(shape=discounted_rewards.shape[0]) * gamma
    gammas = jnp.power(gammas, jnp.array([len(seq) for seq in batch.rewards]))

    discounted_rewards = discounted_rewards + gammas * next_v * ~batch.dones

    target_v = discounted_rewards + gamma * ~batch.dones * next_v
    """
    def value_loss_fn(value_params: Params) -> tuple[Array, dict[str, Array]]:
        # Generate V(s) for the sample states
        layer_outputs, v = value.apply({'params': value_params}, batch.states)

        # Calculate the loss for V using Q with expectile regression
        value_loss = mse_loss(v - target_v)

        # Return the loss value, plus metadata
        return value_loss, {
            'value_loss': value_loss,
            'layer_outputs': layer_outputs,
        }

    # Calculate the updated model parameters using the loss function
    new_value, info = value.apply_gradient(value_loss_fn)

    # Return the new model parameters, plus loss metadata
    return new_value, info

