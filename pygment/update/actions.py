from update.critic import update_q, update_v, update_advantage
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
    actor: Model,
    batch: Batch, **kwargs
) -> tuple[Model, dict[Any, Any]]:

    new_actor, actor_info = update_policy(actor, batch, **kwargs)

    return new_actor, {
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
def _update_advantage_jit(
    advantage: Model, batch: Batch, **kwargs
) -> tuple[Model, dict[Any, Any]]:

    new_advantage, advantage_info = update_advantage(advantage, batch, **kwargs)

    return new_advantage, {
        **advantage_info
    }
