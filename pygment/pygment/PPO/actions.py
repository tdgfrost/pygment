from critic import update_v
from actor import update_policy
from common import Batch

from jax import jit
from jax.random import PRNGKey
import jax

from typing import Any, Tuple, Dict

from net import Model


@jit
def _update_jit(
    rng: PRNGKey, actor: Model, value: Model,
    batch: Batch, gamma: float
) -> tuple[Any, Model, Model, dict[Any, Any]]:

    new_value, value_info = update_v(value, batch)
    key, rng = jax.random.split(rng)

    new_actor, actor_info = update_policy(key, actor, batch)

    return rng, new_actor, new_value, {
        **value_info,
        **actor_info
    }


@jit
def _update_actor_jit(
    rng: PRNGKey, actor: Model, value: Model,
    batch: Batch
) -> tuple[Any, Model, dict[Any, Any]]:

    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_policy(key, actor, value, batch)

    return rng, new_actor, {
        **actor_info
    }


@jit
def _update_value_jit(
    value: Model, batch: Batch,
) -> tuple[Model, dict[Any, Any]]:

    new_value, value_info = update_v(value, batch)

    return new_value, {
        **value_info
    }
