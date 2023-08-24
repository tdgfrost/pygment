from critic import update_q, update_v
from actor import update_policy
from common import Batch

from jax import jit
from jax.random import PRNGKey
import jax

from typing import Any
from net import Model


@jit
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, value: Model,
    batch: Batch, gamma: float, tau: float,
    expectile: float
) -> tuple[Any, Model, Model, Model, dict[Any, Any]]:

    new_value, value_info = update_v(value, batch, expectile)
    key, rng = jax.random.split(rng)

    new_critic, critic_info = update_q(critic, new_value, batch, gamma)

    new_actor, actor_info = update_policy(key, actor, new_critic, new_value, batch)

    return rng, new_actor, new_critic, new_value, {
        **critic_info,
        **value_info,
        **actor_info
    }
