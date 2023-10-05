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
    return weight * jnp.log(diff ** 2 / 2 + 1)


def mc_mse_loss(pred, batch, **kwargs):
    return (pred - batch.discounted_rewards) ** 2


def td_mse_loss(pred, batch, rewards, next_state_values, gamma=0.99, **kwargs):
    gammas = jnp.ones(shape=len(pred)) * gamma
    gammas = jnp.power(gammas, jnp.array([len(traj) for traj in batch.rewards]))

    target = rewards + gammas * next_state_values * (1 - batch.dones)
    return (pred - target) ** 2


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
    adv_filter = nn.relu(jnp.sign(batch.advantages)).astype(jnp.bool_)
    action_logprobs = jnp.where(adv_filter, action_logprobs, 0.)

    # EXPERIMENTAL - try to move towards positive advantages and away from negative advantages
    """
    adv_filter = jnp.exp(5 * batch.advantages)
    action_logprobs *= adv_filter
    """
    # Return the advantage-filtered logprobs
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


def continuous_ranked_probability_score(logits, labels, **kwargs):
    probs = nn.softmax(logits, axis=-1)
    """
    Annoyingly, jnp.cumsum doesn't work on metal backend, so the following code is replaced with
    jax.lax workaround
    # cum_probs = probs.cumsum(-1)
    """
    cum_probs = jax.lax.scan(lambda agg, current: (agg + current, agg + current),
                             jnp.zeros(shape=probs.shape[0]),
                             probs.transpose())[1].transpose()

    label_bool = jnp.arange(cum_probs.shape[-1]).reshape(1, -1) >= labels.reshape(-1, 1)
    """
    crps_score = jnp.absolute(cum_probs - label_bool).mean(-1)
    diff = jnp.expand_dims(cum_probs, -1) - jnp.expand_dims(cum_probs, -2)
    crps_score += -0.5 * jnp.mean(jnp.absolute(diff), axis=(-2, -1))
    """
    return (cum_probs - label_bool) ** 2


def ordinal_crossentropy(logits, labels, **kwargs):
    """
    probs = nn.softmax(logits, axis=-1)
    weight = (jnp.arange(logits.shape[-1]).reshape(1, -1) - labels.reshape(-1, 1) + 1) ** 2
    weight *= probs
    """
    """
    rps = ranked_probability_score(logits, labels, **kwargs)
    rps_weight = rps.sum(1) / (rps.shape[-1] - 1)
    probs = filter_to_action(nn.softmax(logits, -1), labels)
    log_likelihood = -jnp.log(probs)
    log_likelihood = jnp.where(jnp.isinf(log_likelihood),
                               -1000,
                               log_likelihood)
    """
    """
    rps = continuous_ranked_probability_score(logits, labels, **kwargs)
    rps_weight = nn.softmax(rps.sum(-1), -1)

    rps_loss = rps * rps_weight.reshape(-1, 1)
    rps_loss = rps_loss.sum(1)

    probs = filter_to_action(nn.softmax(logits, -1), labels)
    log_likelihood = -jnp.log(probs)
    log_likelihood = jnp.where(jnp.isinf(log_likelihood),
                               -1000,
                               log_likelihood)
    """
    # return rps.sum(1) / (rps.shape[-1] - 1)  # rps_loss * log_likelihood
    """
    probs = nn.softmax(logits, axis=-1)

    lhs_label_bool = jnp.arange(logits.shape[-1]).reshape(1, -1) < labels.reshape(-1, 1)
    rhs_label_bool = jnp.arange(logits.shape[-1]).reshape(1, -1) > labels.reshape(-1, 1)

    def ordinal_crossentropy_loss_fn(reverse=False):
        cum_probs = jax.lax.scan(lambda agg, current: (agg + current, agg + current),
                                 jnp.zeros(shape=probs.shape[0]),
                                 probs.transpose(), reverse=reverse)[1].transpose()

        label_bool = rhs_label_bool if reverse else lhs_label_bool

        crps = (cum_probs - label_bool) ** 2

        mask = jnp.ones(crps.shape[-1])
        mask = jnp.where(label_bool, mask, 0)
        crps *= mask

        ordinal_loss = jnp.log(1 - crps)

        return ordinal_loss.mean(-1)

    lhs_loss = ordinal_crossentropy_loss_fn(reverse=False)
    rhs_loss = ordinal_crossentropy_loss_fn(reverse=True)
    middle_loss = log_softmax_cross_entropy(logits, labels)

    return middle_loss - lhs_loss - rhs_loss
    """
    """
    probs = nn.softmax(logits, axis=-1)

    cum_probs = jax.lax.scan(lambda agg, current: (agg + current, agg + current),
                             jnp.zeros(shape=probs.shape[0]),
                             probs.transpose())[1].transpose()

    label_bool = jnp.arange(cum_probs.shape[-1]).reshape(1, -1) >= labels.reshape(-1, 1)

    # Exclude the last column, where the probability is always 1 (i.e., degrees of freedom is C - 1)
    cum_probs = cum_probs[:, :-1]
    label_bool = label_bool[:, :-1]

    pt = jnp.where(label_bool, cum_probs, 1 - cum_probs)
    loss = -(1 - pt)**2 * jnp.log(pt)

    return loss.sum(-1)
    """
    probs = nn.softmax(logits, axis=-1)
    label_bool = nn.one_hot(labels, logits.shape[-1])

    pt = jnp.where(label_bool, probs, 1 - probs)
    loss = -(1 - pt)**2 * jnp.log(pt)
    
    return loss.sum(-1)

"""
def ordinal_crossentropy(logits, labels, **kwargs):
    probs = 1 / (1 + jnp.exp(-logits))
    label_bool = jnp.arange(probs.shape[-1]).reshape(1, -1) < labels.reshape(-1, 1)

    loss = -(jnp.log(probs) * label_bool + jnp.log(1 - probs) * (1 - label_bool)).sum()

    return loss
"""

"""
def ordinal_crossentropy(logits, labels, **kwargs):
    probs = nn.softmax(logits, axis=-1)
    label_distance = ((jnp.arange(logits.shape[-1]).reshape(1, -1) - labels.reshape(-1, 1) + 1) ** 2) ** 2

    inverse_logprobs = jnp.log(1 - probs)
    inverse_logprobs = jnp.where(jnp.isinf(inverse_logprobs),
                                 -1000,
                                 inverse_logprobs)

    loss = -(inverse_logprobs * label_distance).sum()

    return loss
"""
