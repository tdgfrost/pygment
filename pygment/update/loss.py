import jax
import jax.numpy as jnp
import flax.linen as nn
from core.common import filter_to_action


def convert_logits_to_action_logprobs(logits, actions, **kwargs):
    logprobs = nn.log_softmax(logits, axis=-1)

    """
    This is until jnp.take_along_axis works for metal backend

    action_logprobs = jnp.squeeze(jnp.take_along_axis(action_logprobs,
                                                      actions,
                                                      axis=-1), -1)
    """
    """
    action_logprobs = jax.lax.gather(logprobs,
                                     jnp.concatenate((jnp.arange(len(actions)).reshape(-1, 1),
                                                      actions.reshape(-1, 1)), axis=1),
                                     jax.lax.GatherDimensionNumbers(
                                         offset_dims=tuple([]),
                                         collapsed_slice_dims=tuple([0, 1]),
                                         start_index_map=tuple([0, 1])), tuple([1, 1]))
    """
    action_logprobs = filter_to_action(logprobs, actions)

    action_logprobs = jnp.where(jnp.isinf(action_logprobs),
                                -1000,
                                action_logprobs)

    return action_logprobs


def expectile_loss(pred, batch, expectile=0.8, **kwargs):
    """
    Calculates the expectile of the Cauchy loss function for the residuals.
    Uses the formula L[ c^2/2 * log(loss^2 / c^2 + 1)]
    In this case, c is set to sqrt(2), so the formula simplifies to L[log(loss^2/2 + 1)]

    :param pred: the predicted values
    :param batch: Batch containing the target values
    :param expectile: the expectile value
    :return: the loss term
    """
    diff = pred - batch.discounted_rewards
    weight = jnp.where(diff > 0, (1 - expectile), expectile)
    # return weight * jnp.log(diff ** 2 / 2 + 1)
    return weight * diff ** 2


def mc_mse_loss(pred, batch, **kwargs):
    return (pred - batch.discounted_rewards) ** 2


def iql_loss(logits, batch, **kwargs):
    """
    Calculate the advantage-filtered logprobs

    :param logits: logits from the actor model
    :param batch: the batch of samples
    :param adv_filter: boolean filter for positive advantage actions
    :return: the advantage-filtered logprobs
    """

    # Convert the logits to action log_probs
    action_logprobs = convert_logits_to_action_logprobs(logits, batch.actions)

    # Filter the logprobs using the advantage filter
    """
    adv_filter = nn.relu(jnp.sign(batch.advantages)).astype(jnp.bool_)
    action_logprobs = jnp.where(adv_filter, action_logprobs, 0.)
    """
    """
    adv_filter = batch.advantages > 0
    action_logprobs *= adv_filter
    """
    # EXPERIMENTAL - try to move towards positive advantages and away from negative advantages
    adv_filter = jnp.exp(batch.advantages)
    action_logprobs *= adv_filter

    # Return the advantage-filtered logprobs
    return -action_logprobs


def clone_behaviour(logits, batch, **kwargs):
    """
    Calculate the advantage-filtered logprobs

    :param logits: logits from the actor model
    :param batch: the batch of samples
    :return: the advantage-filtered logprobs
    """

    # Convert the logits to action log_probs
    action_logprobs = convert_logits_to_action_logprobs(logits, batch.actions)

    # Maximise the logprobs
    return -action_logprobs


def ppo_loss(logits, batch, clip_ratio=0.2, **kwargs):
    """
    Calculate the PPO loss function

    :param logits: the logits from the policy network
    :param batch: the batch of samples
    :param clip_ratio: the clipping ratio
    :return: the loss term
    """
    # Unpack batch
    old_logprobs = batch.action_logprobs
    advantage = batch.advantages

    # Get the action logprobs
    logprobs = convert_logits_to_action_logprobs(logits, batch.actions)

    # Calculate the ratio of the new logprobs to the old logprobs
    ratio = jnp.exp(logprobs - old_logprobs)

    # Calculate the clipped ratio
    clipped_ratio = jnp.clip(ratio, 1 - clip_ratio, 1 + clip_ratio)

    # Calculate the minimum of the ratio and the clipped ratio
    min_ratio = jnp.minimum(ratio * advantage, clipped_ratio * advantage)

    # Calculate the PPO loss
    loss = -min_ratio

    # Return the loss term
    return loss


def log_softmax_cross_entropy(logits, labels, **kwargs):
    logits_max = jnp.max(logits, axis=-1, keepdims=True)
    logits -= jax.lax.stop_gradient(logits_max)

    label_logits = filter_to_action(logits, labels)
    log_normalizers = jnp.log(jnp.sum(jnp.exp(logits), axis=-1))

    return log_normalizers - label_logits


def binary_cross_entropy(logits, labels, **kwargs):
    probs = nn.sigmoid(logits)
    log_0 = jnp.where(jnp.isinf(jnp.log(probs)),
                      -1000,
                      jnp.log(probs))
    log_1 = jnp.where(jnp.isinf(jnp.log(1 - probs)),
                      -1000,
                      jnp.log(1 - probs))
    loss = labels * log_0 + (1 - labels) * log_1
    return -loss

