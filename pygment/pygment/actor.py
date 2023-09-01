from typing import Tuple, Dict

import jax.numpy as jnp
import flax.linen as nn
from jax import Array
import jax

from common import Batch, InfoDict, Params, PRNGKey
from agent import Model


def loss(logits, actions, adv_filter):
    """
    Calculate the advantage-filtered logprobs

    :param logits: logits from the actor model
    :param actions: sample actions
    :param adv_filter: boolean filter for positive advantage actions
    :return: the advantage-filtered logprobs
    """

    # Convert the logits to log_probs, and select the log_probs for the sampled actions
    action_logprobs = nn.log_softmax(logits, axis=-1)
    """
    This is until jnp.take_along_axis works for metal backend
    
    action_logprobs = jnp.squeeze(jnp.take_along_axis(action_logprobs,
                                                      actions,
                                                      axis=-1), -1)
    """
    action_logprobs = jax.lax.gather(action_logprobs,
                                     jnp.concatenate((jnp.arange(len(actions)).reshape(-1, 1), actions), axis=1),
                                     jax.lax.GatherDimensionNumbers(
                                         offset_dims=tuple([]),
                                         collapsed_slice_dims=tuple([0, 1]),
                                         start_index_map=tuple([0, 1])), tuple([1, 1]))

    # Remove any NaNs / infinite values
    action_logprobs = jnp.where(jnp.isinf(action_logprobs) | jnp.isnan(action_logprobs),
                                -1000,
                                action_logprobs)
    # action_logprobs = jnp.where(action_logprobs == 0, -1e-8, action_logprobs)

    # Filter the logprobs using the advantage filter
    action_logprobs = jnp.where(adv_filter, action_logprobs, 0.)

    # Return the advantage-filtered logprobs
    return action_logprobs


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

    # Calculate the Q(s,a) for the states and actions in the batch
    _, (q1, q2) = critic(batch.states)
    q = jnp.minimum(q1, q2)
    """
    This is until jnp.take_along_axis works for metal backend.
    
    q = jnp.squeeze(jnp.take_along_axis(q, batch.actions, axis=-1), -1)
    """
    q = jax.lax.gather(q,
                       jnp.concatenate((jnp.arange(len(batch.actions)).reshape(-1, 1), batch.actions), axis=1),
                       jax.lax.GatherDimensionNumbers(
                           offset_dims=tuple([]),
                           collapsed_slice_dims=tuple([0, 1]),
                           start_index_map=tuple([0, 1])), tuple([1, 1]))

    # Calculate the advantages (with a boolean filter for positive advantages)
    adv_filter = nn.relu(jnp.sign(q - v)).astype(jnp.bool_)

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

        # Convert this to advantage-filtered logprobs
        action_logprobs = loss(logits, batch.actions, adv_filter)

        # Calculate the loss for the actor model
        actor_loss = -action_logprobs.mean()

        # Return the loss value, plus metadata
        return actor_loss, {'actor_loss': actor_loss,
                            'layer_outputs': layer_outputs,
                            }

    # Calculate the updated model parameters using the loss function
    new_actor, info = actor.apply_gradient(actor_loss_fn)

    # Return the new model parameters, plus loss metadata
    return new_actor, info
