from typing import Tuple, Dict

import jax.numpy as jnp
import flax.linen as nn
from jax import Array
import jax

from common import Batch, InfoDict, Params, PRNGKey
from agent import Model


def ppo_loss(logprobs, old_logprobs, advantage, clip_ratio=0.2):
    """
    Calculate the PPO loss function

    :param logprobs: the logprobs of the actions taken
    :param old_logprobs: the logprobs of the actions taken in the previous iteration
    :param advantage: the advantage of the actions taken
    :param clip_ratio: the clipping ratio
    :return: the loss term
    """

    # Calculate the ratio of the new logprobs to the old logprobs
    ratio = jnp.exp(logprobs - old_logprobs)

    # Calculate the clipped ratio
    clipped_ratio = jnp.clip(ratio, 1 - clip_ratio, 1 + clip_ratio)

    # Calculate the minimum of the ratio and the clipped ratio
    min_ratio = jnp.minimum(ratio * advantage, clipped_ratio * advantage)

    # Calculate the PPO loss
    loss = -jnp.mean(min_ratio)

    # Return the loss term
    return loss


def update_policy(key: PRNGKey, actor: Model, critic: Model, value: Model,
                  batch: Batch) -> Tuple[Model, InfoDict]:
    """
    Update the policy using the advantage-filtered logprobs

    :param key: JAX random key
    :param actor: The actor model
    :param critic: The critic model
    :param value: The value model
    :param batch: a Batch object of samples
    :return: a tuple containing the new model parameters, plus metadata
    """

    # Calculate the optimal V(s') for the states in the batch
    _, v = value(batch.states)

    # Calculate the Advantage
    advantage = batch.rewards - v

    # Get old action logprobs
    old_logprobs = batch.action_logprobs

    """
    q = jax.lax.gather(q,
                       jnp.concatenate((jnp.arange(len(batch.actions)).reshape(-1, 1), batch.actions), axis=1),
                       jax.lax.GatherDimensionNumbers(
                           offset_dims=tuple([]),
                           collapsed_slice_dims=tuple([0, 1]),
                           start_index_map=tuple([0, 1])), tuple([1, 1]))
    """

    def actor_loss_fn(actor_params: Params) -> tuple[Array, dict[str, Array]]:
        """
        Calculate the loss for the actor model

        :param actor_params: the parameters of the actor model
        :return: a tuple containing the loss value, plus metadata
        """

        # Generate the logits for the actions
        layer_outputs, logits = actor.apply({'params': actor_params},
                                            batch.states,
                                            training=True,
                                            rngs={'dropout': key})

        # Convert this to logprobs
        action_logprobs = nn.log_softmax(logits)

        # Calculate the loss for the actor model
        actor_loss = ppo_loss(action_logprobs, old_logprobs, advantage, clip_ratio=0.2)

        # Return the loss value, plus metadata
        return actor_loss, {'actor_loss': actor_loss,
                            'layer_outputs': layer_outputs,
                            }

    # Calculate the updated model parameters using the loss function
    new_actor, info = actor.apply_gradient(actor_loss_fn)

    for _ in range(4):
        new_actor, info = new_actor.apply_gradient(actor_loss_fn)

    # Return the new model parameters, plus loss metadata
    return new_actor, info
