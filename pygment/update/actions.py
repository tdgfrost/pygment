from update.critic import update_q, update_v
from update.actor import update_policy
from core.common import Batch

from jax import jit
from jax.random import PRNGKey
import jax

from typing import Any, Tuple, Dict

from core.net import Model


@jit
def _update_jit(
    actor: Model = None, critic: Model = None, value: Model = None,
    batch: Batch = None, **kwargs
) -> tuple[Any, Any, Any, dict[Any, Any]]:

    new_value, value_info = update_v(value, batch, **kwargs) if value is not None else (value, {})

    new_critic, critic_info = update_q(critic, batch, **kwargs) if critic is not None else (critic, {})

    new_actor, actor_info = update_policy(actor, batch, **kwargs) if actor is not None else (actor, {})

    return new_actor, new_critic, new_value, {
        **critic_info,
        **value_info,
        **actor_info
    }


@jit
def _update_actor_jit(
        key: PRNGKey, actor: Model, batch: Batch, **kwargs
) -> tuple[Any, Model, dict[Any, Any]]:

    key, rng = jax.random.split(key)
    new_actor, actor_info = update_policy(rng, actor, batch, **kwargs)

    return key, new_actor, {
        **actor_info
    }


@jit
def _update_critic_jit(
    key: PRNGKey, critic: Model, batch: Batch, **kwargs
) -> tuple[Any, Model, dict[Any, Any]]:

    key, rng = jax.random.split(key)
    new_critic, critic_info = update_q(rng, critic, batch, **kwargs)

    return key, new_critic, {
        **critic_info
    }


@jit
def _update_value_jit(
    key: PRNGKey, value: Model, batch: Batch, **kwargs
) -> tuple[Any, Model, dict[Any, Any]]:

    key, rng = jax.random.split(key)
    new_value, value_info = update_v(rng, value, batch, **kwargs)

    return key, new_value, {
        **value_info
    }

