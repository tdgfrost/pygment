from update.critic import update_q, update_v, update_uncertainty
from update.actor import update_policy
from core.common import Batch

from jax import jit
from jax.random import PRNGKey
import jax

from typing import Any

from core.net import Model


@jit
def _update_jit(
    rng: PRNGKey, actor: Model = None, critic: Model = None, value: Model = None,
    batch: Batch = None, **kwargs
) -> tuple[Any, Model, Model, Model, dict[Any, Any]]:

    new_value, value_info = update_v(value, batch, **kwargs) if value is not None else (value, {})
    key, rng = jax.random.split(rng)

    new_critic, critic_info = update_q(critic, batch, **kwargs) if critic is not None else (critic, {})

    new_actor, actor_info = update_policy(actor, batch, **kwargs) if actor is not None else (actor, {})

    return rng, new_actor, new_critic, new_value, {
        **critic_info,
        **value_info,
        **actor_info
    }


@jit
def _update_actor_jit(
    rng: PRNGKey, actor: Model,
    batch: Batch, **kwargs
) -> tuple[Any, Model, dict[Any, Any]]:

    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_policy(actor, batch, **kwargs)

    return rng, new_actor, {
        **actor_info
    }


@jit
def _update_critic_jit(
    critic: Model, batch: Batch, **kwargs
) -> tuple[Model, dict[Any, Any]]:

    new_critic, critic_info = update_q(critic, batch, **kwargs)

    return new_critic, {
        **critic_info
    }


@jit
def _update_value_jit(
    value: Model, batch: Batch, **kwargs
) -> tuple[Model, dict[Any, Any]]:

    new_value, value_info = update_v(value, batch, **kwargs)

    return new_value, {
        **value_info
    }



@jit
def _update_uncertainty_jit(
    uncertainty: Model, batch: Batch, **kwargs
) -> tuple[Model, dict[Any, Any]]:

    new_uncertainty, uncertainty_info = update_uncertainty(uncertainty, batch, **kwargs)

    return new_uncertainty, {
        **uncertainty_info
    }
