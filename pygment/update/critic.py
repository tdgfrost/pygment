from jax import Array

from core.agent import Model
from core.common import Params, InfoDict, Batch, filter_to_action, split_output
from update.loss import mc_mse_loss, expectile_loss, gaussian_mse_loss, gaussian_expectile_loss, gaussian_nll_loss

from typing import Tuple


def update_v(value: Model, batch: Batch, **kwargs) -> Tuple[Model, InfoDict]:
    """
    Function to update the Value network

    :param value: the Value network to be updated
    :param batch: a Batch object of samples
    :return: a tuple containing the new model parameters, plus metadata
    """

    loss_fn = {'mc_mse': mc_mse_loss,
               'expectile': expectile_loss,
               'gaussian_nll': gaussian_nll_loss,
               'gaussian_mse': gaussian_mse_loss,
               'gaussian_expectile': gaussian_expectile_loss}

    # Unpack the actions, states, and discounted rewards from the batch of samples
    states = batch.states

    def value_loss_fn(value_params: Params) -> tuple[Array, dict[str, Array]]:
        # Generate V(s) for the sample states
        layer_outputs, v = value.apply({'params': value_params}, states)

        v_mu, v_std = split_output(v)

        if batch.len_actions is not None and len(v_mu.shape) > 1:
            v_mu = filter_to_action(v_mu, batch.len_actions)
            v_std = filter_to_action(v_std, batch.len_actions)

        # Calculate the loss for V using Q with expectile regression
        value_loss = loss_fn[list(kwargs['value_loss_fn'].keys())[0]]((v_mu, v_std), batch, **kwargs).mean()

        # Return the loss value, plus metadata
        return value_loss, {
            'value_loss': value_loss,
            'layer_outputs': layer_outputs,
        }

    # Calculate the updated model parameters using the loss function
    new_value, info = value.apply_gradient(value_loss_fn)

    # Return the new model parameters, plus loss metadata
    return new_value, info


def update_q(critic: Model, batch: Batch, **kwargs) -> Tuple[Model, InfoDict]:
    """
    Function to update the Q network
    :param critic: the critic network to be updated
    :param batch: a Batch object of samples
    :return: a tuple containing the new model parameters, plus metadata
    """

    # Unpack the actions, states, and discounted rewards from the batch of samples
    actions = batch.actions
    states = batch.states

    loss_fn = {'mc_mse': mc_mse_loss,
               'expectile': expectile_loss,
               'gaussian_mse': gaussian_mse_loss,
               'gaussian_expectile': gaussian_expectile_loss}

    def critic_loss_fn(critic_params: Params) -> tuple[Array, dict[str, Array]]:
        # Generate Q values from each of the two critic networks
        layer_outputs, (q1, q2) = critic.apply({'params': critic_params}, states)

        # Select the sampled actions
        """
        This is just until jnp.take_along_axis works on metal again
        
        q1 = jnp.squeeze(jnp.take_along_axis(q1, actions, axis=-1), -1)
        q2 = jnp.squeeze(jnp.take_along_axis(q2, actions, axis=-1), -1)
        """
        (q1_mu, q1_std), (q2_mu, q2_std) = split_output(q1), split_output(q2)

        q1_mu, q1_std = filter_to_action(q1_mu, actions), filter_to_action(q1_std, actions)
        q2_mu, q2_std = filter_to_action(q2_mu, actions), filter_to_action(q2_std, actions)

        # Calculate the loss for the critic networks using the target Q values with MSE
        critic_loss_1 = loss_fn[list(kwargs['critic_loss_fn'].keys())[0]]((q1_mu, q2_std), batch, **kwargs).mean()
        critic_loss_2 = loss_fn[list(kwargs['critic_loss_fn'].keys())[0]]((q2_mu, q2_std), batch, **kwargs).mean()
        critic_loss = critic_loss_1 + critic_loss_2

        # Return the loss value, plus metadata
        return critic_loss, {
            'critic_loss': critic_loss,
            'layer_outputs': layer_outputs,
        }

    # Calculate the updated model parameters using the loss function
    new_critic, info = critic.apply_gradient(critic_loss_fn)

    # Return the new model parameters, plus loss metadata
    return new_critic, info

