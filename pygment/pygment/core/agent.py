from core.net import Model, ValueNet, ActorNet, DoubleCriticNet, CriticNet
from core.common import Batch, InfoDict, alter_batch
from update.actions import _update_jit, _update_value_jit, _update_critic_jit, _update_actor_jit, _update_interval_jit

import os
import datetime as dt

import numpy as np
import jax.numpy as jnp
import jax
from jax import jit
import flax.linen as nn
import optax

from typing import List, Optional, Sequence, Dict


class BaseAgent:
    """
    Base class for all agents. Contains the basic functions that all agents should have.
    """

    def __init__(self,
                 path=None):
        """
        Creates a unique folder associated with the current agent.

        :param path: optional path to save the agent to.
        """
        now = dt.datetime.now()
        now = f'{now.year}_{now.month}_{now.day:02}_{now.hour:02}{now.minute:02}{now.second:02}'
        if path is None:
            self.path = now
        else:
            self.path = os.path.join(path, now)

        self.networks = []

    @staticmethod
    def sample(data,
               batch_size):
        """
        Samples a batch from the data.

        :param data: data in the form of a Batch object.
        :param batch_size: desired batch size.
        :return: a Batch object of size batch_size.
        """
        idxs = np.random.default_rng().choice(len(data.dones),
                                              size=batch_size,
                                              replace=False)

        batch = data._asdict()

        for key, val in batch.items():
            if val is None:
                continue
            if key == 'rewards':
                batch[key] = [val[i] for i in idxs]
            elif key == 'episode_rewards':
                batch[key] = val
            else:
                batch[key] = val[idxs]

        return Batch(**batch)

    def standardise_inputs(self, inputs: np.ndarray):
        """
        Standardises the inputs to the networks.

        :param inputs: inputs to the networks.
        :return: standardised inputs.
        """
        for network in self.networks:
            network.__dict__['input_mean'] = np.mean(inputs, axis=0)
            network.__dict__['input_std'] = np.maximum(np.std(inputs, axis=0), 1e-8)


class IQLAgent(BaseAgent):
    """
    IQL Agent.
    """

    def __init__(self,
                 seed: int,
                 observations: np.ndarray,
                 action_dim: int,
                 interval_dim: int,
                 interval_min: int,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 interval_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 gamma: float = 0.99,
                 expectile: float = 0.8,
                 epochs: Optional[int] = None,
                 opt_decay_schedule: str = "cosine",
                 clipping: float = 0.01,
                 continual_learning: bool = False,
                 *args,
                 **kwargs):

        # Initiate the BaseAgent characteristics
        super().__init__()

        # Set random seed
        rng = jax.random.PRNGKey(seed)
        self.rng, self.actor_key, self.critic_key, self.interval_key, self.value_key = jax.random.split(rng, 5)

        # Set parameters
        self.action_dim = action_dim
        self.interval_dim = interval_dim

        # Set hyperparameters
        self.expectile = expectile
        self.gamma = gamma

        # Set optimizers
        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, epochs)
            optimiser = optax.chain(optax.clip_by_global_norm(clipping),
                                    optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            optimiser = optax.adam(learning_rate=actor_lr)

        """
        Each Model essentially contains a neural network structure, parameters, and optimiser.
        The parameters are kept separate from the neural network structure, and updated separately.
        """
        # Set models
        self.actor = Model.create(ActorNet(hidden_dims, self.action_dim),
                                  inputs=[self.actor_key, observations],
                                  optim=optimiser,
                                  continual_learning=continual_learning)

        self.critic = Model.create(DoubleCriticNet(hidden_dims, self.action_dim),
                                   inputs=[self.critic_key, observations],
                                   optim=optax.adam(learning_rate=critic_lr),
                                   continual_learning=continual_learning)

        self.interval = Model.create(CriticNet(hidden_dims, self.interval_dim),
                                     inputs=[self.interval_key, observations],
                                     optim=optax.adam(learning_rate=interval_lr),
                                     continual_learning=continual_learning)

        self.value = Model.create(ValueNet(hidden_dims, self.interval_dim),
                                  inputs=[self.value_key, observations],
                                  optim=optax.adam(learning_rate=value_lr),
                                  continual_learning=continual_learning)

        self.networks = [self.actor, self.critic, self.value]

    def update(self, batch: Batch, **kwargs) -> InfoDict:
        """
        Updates the agent's networks.

        :param batch: a Batch object.
        :return: an InfoDict object containing metadata.
        """

        # Create an updated copy of all the networks
        new_rng, new_actor, new_critic, new_value, info = _update_jit(
            self.rng, self.actor, self.critic, self.value,
            batch, **kwargs)

        # Update the agent's networks with the updated copies
        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value

        # Return the metadata
        return info

    def update_async(self, batch: Batch, interval:bool = False, actor: bool = False,
                     critic: bool = False, value: bool = False, **kwargs) -> InfoDict:
        """
        Updates the agent's networks asynchronously.

        :param batch: a Batch object.
        :param actor: whether to update the actor network.
        :param critic: whether to update the critic network.
        :param value: whether to update the value network.
        :return: an InfoDict object containing metadata.
        """

        # Create an updated copy of the required networks
        new_interval, interval_info = _update_interval_jit(
            self.interval, batch, **kwargs) if interval else (self.interval, {}
        )

        new_rng, new_actor, actor_info = _update_actor_jit(
            self.rng, self.actor, batch, **kwargs) if actor else (self.rng, self.actor, {})

        new_critic, critic_info = _update_critic_jit(
            self.critic, batch, **kwargs) if critic else (self.critic, {})

        new_value, value_info = _update_value_jit(
            self.value, batch, **kwargs) if value else (self.value, {})

        # Update the agent's networks with the updated copies
        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.interval = new_interval

        # Return the metadata
        return {
            **interval_info,
            **critic_info,
            **value_info,
            **actor_info
        }

    def sample_action(self, state, key=jax.random.PRNGKey(123)):
        """
        Chooses an action based on the current state.

        :param state: the current state.
        :param key: a PRNGKey.
        :return: an action (as an integer if a single state, or an Array if multiple states)
        """

        _, logits = self.actor(state)

        action, logprobs = self._sample_action_jit(logits, key)
        return np.array(action), np.array(logprobs)

    @staticmethod
    @jit
    def _sample_action_jit(logits, key):
        log_probs = nn.log_softmax(logits, axis=-1)
        action = jax.random.categorical(key, logits, axis=-1)
        return action, log_probs


class PPOAgent(BaseAgent):
    """
    IQL Agent.
    """

    def __init__(self,
                 seed: int,
                 observations: np.ndarray,
                 action_dim: int,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 gamma: float = 0.99,
                 epochs: Optional[int] = None,
                 opt_decay_schedule: str = "cosine",
                 clipping: float = 0.01,
                 *args,
                 **kwargs):

        # Initiate the BaseAgent characteristics
        super().__init__()

        # Set random seed
        rng = jax.random.PRNGKey(seed)
        self.rng, self.actor_key, self.value_key = jax.random.split(rng, 3)

        # Set parameters
        self.action_dim = action_dim

        # Set hyperparameters
        self.gamma = gamma

        # Set optimizers
        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, epochs)
            optimiser = optax.chain(optax.clip_by_global_norm(clipping),
                                    optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            optimiser = optax.adam(learning_rate=actor_lr)

        """
        Each Model essentially contains a neural network structure, parameters, and optimiser.
        The parameters are kept separate from the neural network structure, and updated separately.
        """
        # Set models
        self.actor = Model.create(ActorNet(hidden_dims, self.action_dim),
                                  inputs=[self.actor_key, observations],
                                  optim=optimiser)

        self.value = Model.create(ValueNet(hidden_dims),
                                  inputs=[self.value_key, observations],
                                  optim=optax.adam(learning_rate=value_lr))

        self.networks = [self.actor, self.value]

    def update(self, batch: Batch, **kwargs) -> InfoDict:
        """
        Updates the agent's networks.

        :param batch: a Batch object.
        :return: an InfoDict object containing metadata.
        """

        # Create an updated copy of all the networks
        new_rng, new_actor, _, new_value, info = _update_jit(
            rng=self.rng,
            actor=self.actor,
            value=self.value,
            batch=batch,
            **kwargs)

        # Update the agent's networks with the updated copies
        self.rng = new_rng
        self.actor = new_actor
        self.value = new_value

        # Return the metadata
        return info

    def sample_action(self, state, key=jax.random.PRNGKey(123)):
        """
        Chooses an action based on the current state.

        :param state: the current state.
        :param key: a PRNGKey.
        :return: an action (as an integer if a single state, or an Array if multiple states)
        """

        _, logits = self.actor(state)

        action, logprobs = self._sample_action_jit(logits, key)
        return np.array(action), np.array(logprobs)

    @staticmethod
    @jit
    def _sample_action_jit(logits, key):
        log_probs = nn.log_softmax(logits, axis=-1)
        action = jax.random.categorical(key, logits, axis=-1)
        return action, log_probs
