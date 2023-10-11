from typing import Tuple

from jax import Array

from core.common import Batch, InfoDict, Params
from core.agent import Model
from update.loss import ppo_loss, iql_loss, clone_behaviour


def update_policy(actor: Model, batch: Batch, **kwargs) -> Tuple[Model, InfoDict]:
    """
    Update the policy using the advantage-filtered logprobs

    :param actor: The actor model
    :param batch: a Batch object of samples
    :return: a tuple containing the new model parameters, plus metadata
    """

    loss_fn = {'ppo': ppo_loss,
               'iql': iql_loss,
               'clone': clone_behaviour}

    def actor_loss_fn(actor_params: Params) -> tuple[Array, dict[str, Array]]:
        """
        Calculate the loss for the actor model

        :param actor_params: the parameters of the actor model
        :return: a tuple containing the loss value, plus metadata
        """

        # Generate the logits for the actions
        layer_outputs, logits = actor.apply({'params': actor_params},
                                            batch.states)

        # Convert this to advantage-filtered logprobs - instead of overall mean, we take the 'mean' of the valid actions
        """
        num_of_valid_actions = jnp.sum(nn.relu(jnp.sign(batch.advantages)).astype(jnp.bool_))
        actor_loss = loss_fn[list(kwargs['actor_loss_fn'].keys())[0]](logits, 
                                                                        batch, **kwargs).sum() / num_of_valid_actions
        """
        # EXPERIMENTAL (also need to update loss function)
        # - try to move towards positive advantages and away from negative advantages
        actor_loss = loss_fn[list(kwargs['actor_loss_fn'].keys())[0]](logits,
                                                                      batch,
                                                                      **kwargs).mean()

        # Return the loss value, plus metadata
        return actor_loss, {'actor_loss': actor_loss,
                            'layer_outputs': layer_outputs,
                            }

    # Calculate the updated model parameters using the loss function
    new_actor, info = actor.apply_gradient(actor_loss_fn)

    # Return the new model parameters, plus loss metadata
    return new_actor, info
