from net import Model, ValueNet, ActorNet, DoubleCriticNet
from common import Batch, InfoDict
from actions import _update_jit

import os
import datetime as dt

import numpy as np
import jax.numpy as jnp
import jax
import optax

from typing import List, Optional, Sequence, Dict


class BaseAgent:
    def __init__(self,
                 path=None):
        now = dt.datetime.now()
        now = f'./{now.year}_{now.month}_{now.day:02}_{now.hour:02}{now.minute:02}{now.second:02}'
        if path is None:
            self.path = now
        else:
            self.path = os.path.join(path, now.lstrip('./'))

    @staticmethod
    def sample(data,
               batch_size):
        idxs = np.random.default_rng().choice(len(data),
                                              size=batch_size,
                                              replace=False)
        return Batch(states=data.state[idxs],
                     actions=data.action[idxs],
                     rewards=data.reward[idxs],
                     discounted_rewards=data.discounted_reward[idxs],
                     next_states=data.next_state[idxs],
                     next_actions=data.next_action[idxs],
                     dones=data.done[idxs])


class IQLAgent(BaseAgent):
    """
    Implementation of IQL
    """

    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 gamma: float = 0.99,
                 tau: float = 0.5,
                 expectile: float = 0.8,
                 dropout_rate: Optional[float] = None,
                 epochs: Optional[int] = None,
                 opt_decay_schedule: str = "cosine",
                 *args,
                 **kwargs):
        super().__init__()

        # Set random seed
        rng = jax.random.PRNGKey(seed)
        self.rng, self.actor_key, self.critic_key, self.value_key = jax.random.split(rng, 4)

        # Set parameters
        action_dim = actions.shape[-1]

        # Set hyperparameters
        self.expectile = expectile
        self.tau = tau
        self.gamma = gamma

        # Set optimizers
        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, epochs)
            optimiser = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            optimiser = optax.adam(learning_rate=actor_lr)

        # Set models
        self.actor = Model.create(ActorNet(hidden_dims),
                                  inputs=[self.actor_key, observations],
                                  tx=optimiser)

        self.critic = Model.create(DoubleCriticNet(hidden_dims),
                                   inputs=[self.critic_key, observations, actions],
                                   tx=optax.adam(learning_rate=critic_lr))

        self.target_critic = Model.create(DoubleCriticNet(hidden_dims),
                                          inputs=[self.critic_key, observations, actions],
                                          tx=optax.adam(learning_rate=critic_lr))

        self.value = Model.create(ValueNet(hidden_dims),
                                  inputs=[self.value_key, observations],
                                  tx=optax.adam(learning_rate=value_lr))

    def update(self,
               batch: Batch,
               **kwargs) -> InfoDict:
        new_rng, new_actor, new_critic, new_value, new_target_critic, info = _update_jit(
            self.rng, self.actor, self.critic, self.value, self.target_critic,
            batch, self.gamma, self.tau, self.expectile)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_critic = new_target_critic

        return info


"""

class IQLAgent(BaseAgent):
    '''
    IQLAgent is an expansion of IQLAgent, allowing for variable lengths between decisions.
    '''

    def __init__(self, device):
        super().__init__(device)
        self._alpha = None
        self.custom_params = None
        self._beta = None
        self._tau = None
        self.value = CriticNet()
        self.critic = DualNet()
        self.actor = ActorNet()
        self.behaviour_policy = ActorNet()
        self.net = self.critic1
        self._batch_size = None

    def add_network(self, nodes: list):
        super().network_check(nodes)

        self.value.add_layers(nodes, self.env)
        self.critic.add_layers(nodes, self.env)
        self.actor.add_layers(nodes, self.env)
        self.behaviour_policy.add_layers(nodes, self.env)
        self.value.has_net = True
        self.critic.has_net = True
        self.actor.has_net = True
        self.behaviour_policy.has_net = True

    def compile(self, optimizer, learning_rate=0.001, weight_decay=1e-5, clip=1.0, lower_clip=None, upper_clip=None):
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._regularisation = weight_decay
        super().compile_check()

        self.value.to(self.device)
        self.critic.main_net.to(self.device)
        self.critic.target_net.to(self.device)
        self.actor.to(self.device)
        self.behaviour_policy.to(self.device)
        '''
        for params in [self.value.parameters(), self.critic1.main_net.parameters(), self.critic2.main_net.parameters(),
                       self.critic1.target_net.parameters(), self.critic2.target_net.parameters(),
                       self.actor.parameters(), self.behaviour_policy.parameters()]:
        '''
        for params in [self.value.parameters(), self.critic.main_net.parameters(), self.critic.target_net.parameters(),
                       self.actor.parameters(),
                       self.behaviour_policy.parameters()]:
            for p in params:
                p.register_hook(lambda grad: torch.clamp(grad,
                                                         lower_clip if lower_clip is not None else -clip,
                                                         upper_clip if upper_clip is not None else clip))

        self._compiled = True

    @staticmethod
    def sample(data, batch_size):
        idxs = np.random.default_rng().choice(len(data), size=batch_size, replace=False)
        return [data[idx] for idx in idxs]

    def clone_behaviour(self, data, batch_size=64, epochs=10000, evaluate=False, save=False):

        '''
        Draft - need to check if any changes are required to make this actually function correctly.
        '''

        # Set up optimiser
        self.custom_params = []
        self.custom_params.append({'params': self.behaviour_policy.parameters(),
                                   'lr': self._learning_rate,
                                   'weight_decay': self._regularisation})

        super().train_base(0.99, custom_params=self.custom_params)

        # Make save directory if needed
        if save:
            if not os.path.isdir(self.path):
                os.makedirs(self.path)

        # Create logs
        old_policy_loss = torch.inf
        best_policy_loss = torch.inf
        current_loss_policy = []

        # Start training
        print('Beginning training...\n')
        progress_bar = tqdm(range(1, int(epochs) + 1), file=sys.stdout)
        for epoch in progress_bar:
            batch = self.sample(data, batch_size)

            loss = self._update_behaviour_policy(batch)

            current_loss_policy.append(loss)

            if epoch % 1 == 0:
                if evaluate:
                    _, _, _, _, total_rewards = self.evaluate(episodes=1000, parallel_envs=512,
                                                              verbose=False, behaviour=True)

                    with open(
                            '/Users/thomasfrost/Documents/Github/pygment/Informal experiments/clone_behaviour/all_rewards.txt',
                            'a') as f:
                        f.write(f'{int(np.array(total_rewards.mean()))} ')
                        f.close()

                    with open(
                            '/Users/thomasfrost/Documents/Github/pygment/Informal experiments/clone_behaviour/policy_loss.txt',
                            'a') as f:
                        f.write(f'{round(np.array(current_loss_policy).mean(), 6)} ')
                        f.close()

                print(f'\nSteps completed: {epoch}\n')
                print(
                    f'Behaviour policy loss {"decreased" if np.array(current_loss_policy).mean() < old_policy_loss else "increased"} '
                    f'from {old_policy_loss} to {round(np.array(current_loss_policy).mean(), 5)}'
                )
                print(f'Best policy loss: {min(best_policy_loss, round(np.array(current_loss_policy).mean(), 5))}')
                if evaluate:
                    print(f'Average reward: {int(np.array(total_rewards).mean())}')

                if np.array(current_loss_policy).mean() < best_policy_loss:
                    if save:
                        for net, name in [
                            [self.behaviour_policy, 'behaviour_policy']
                        ]:
                            old_save_path = os.path.join(self.path, f'{name}_{best_policy_loss}.pt')
                            new_save_path = os.path.join(self.path,
                                                         f'{name}_{round(np.array(current_loss_policy).mean(), 5)}.pt')

                            torch.save(net, new_save_path)
                            if os.path.isfile(old_save_path) and old_save_path != new_save_path:
                                os.remove(old_save_path)

                    best_policy_loss = round(np.array(current_loss_policy).mean(), 5)

                old_policy_loss = round(np.array(current_loss_policy).mean(), 5)
                current_loss_policy = []

    def train(self, data, critic=True, value=True, actor=True, evaluate=True, steps=1000, batch_size=64,
              gamma=0.99, tau=0.99, beta=1, update_iter=4, ppo_clip=0.01, ppo_clip_decay=0.9, save=False):
        '''
        Variable 'data' should be a 1D list of Experiences - sorted or unsorted. The reward value should be the
        correct Q_s value for that state i.e., the cumulated discounted reward from that state onwards.
        '''
        if save:
            evaluate = True

        # Set up optimiser
        self.method_check(env_loaded=True, net_exists=True, compiled=True)
        self._gamma = gamma

        self.optimizer = {}
        scheduler = {}

        for network_name, network in [
            ['value', self.value],
            ['critic_main', self.critic.main_net],
            ['critic_target', self.critic.target_net],
            ['actor', self.actor]
        ]:
            self.optimizer[network_name] = self._optimizers[self._optimizer](network.parameters(),
                                                                             lr=self._learning_rate,
                                                                             weight_decay=self._regularisation)

            scheduler[network_name] = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer[network_name],
                                                                                 T_max=steps)

        # Create stochastic weight averaged models
        swa_critic_main_net = torch.optim.swa_utils.AveragedModel(self.critic.main_net)
        swa_critic_target_net = torch.optim.swa_utils.AveragedModel(self.critic.target_net)
        swa_value = torch.optim.swa_utils.AveragedModel(self.value)

        swa_scheduler = {}
        for optim_name in ['critic_main', 'critic_target', 'value']:
            swa_scheduler[optim_name] = torch.optim.swa_utils.SWALR(self.optimizer[optim_name], swa_lr=0.01)

        # Make save directory if needed
        if save:
            if not os.path.isdir(self.path):
                os.makedirs(self.path)

        # Create logs
        old_qt_loss = torch.inf
        old_v_loss = torch.inf
        old_policy_loss = torch.inf
        old_average_reward = -10 ** 10
        current_loss_qt = []
        current_loss_v = []
        current_loss_policy = []
        current_best_reward = -10 ** 10

        # If evaluating, start ray instance
        # if evaluate:
        # ray.init()

        # Start training
        print('Beginning training...\n')
        progress_bar = tqdm(range(1, int(steps) + 1), file=sys.stdout)
        for step in progress_bar:
            batch = self.sample(data, batch_size)

            loss_qt = self._update_q(batch, gamma) if critic else None
            loss_v = self._update_v(batch, tau) if value else None

            current_loss_qt.append(loss_qt)
            current_loss_v.append(loss_v)

            if step > 50 and step % 5 == 0:
                swa_critic_main_net.update_parameters(self.critic.main_net)
                swa_critic_target_net.update_parameters(self.critic.target_net)
                swa_value.update_parameters(self.value)
                for optim_name in ['critic_main', 'critic_target', 'value']:
                    swa_scheduler[optim_name].step()

                loss_policy = self._update_policy(batch, beta, update_iter, ppo_clip) if actor else None
                current_loss_policy.append(loss_policy)
                ppo_clip *= ppo_clip_decay

            else:
                for network_name in ['value', 'critic_main', 'critic_target', 'actor']:
                    scheduler[network_name].step()

            if step % 100 == 0:

                if evaluate:
                    _, _, _, _, total_rewards = self.evaluate(episodes=1000, parallel_envs=512,
                                                              verbose=False)

                    with open(
                            '/Users/thomasfrost/Documents/Github/pygment/Informal experiments/cauchy_loss/all_rewards.txt',
                            'a') as f:
                        f.write(f'{int(np.array(total_rewards.mean()))} ')
                        f.close()

                else:
                    total_rewards = np.array([0])

                print(f'\nSteps completed: {step}\n')
                if critic:
                    print(
                        f'Qt_loss {"decreased" if np.array(current_loss_qt).mean() < old_qt_loss else "increased"} '
                        f'from {round(old_qt_loss, 5)} to {round(np.array(current_loss_qt).mean(), 5)}'
                    )
                if value:
                    print(
                        f'V_loss {"decreased" if np.array(current_loss_v).mean() < old_v_loss else "increased"} '
                        f'from {round(old_v_loss, 5)} to {round(np.array(current_loss_v).mean(), 5)}'
                    )
                if actor:
                    print(
                        f'Policy loss {"decreased" if np.array(current_loss_policy).mean() < old_policy_loss else "increased"} '
                        f'from {round(old_policy_loss, 5)} to {round(np.array(current_loss_policy).mean(), 5)}'
                    )
                if evaluate:
                    print(
                        f'Average reward {"decreased" if total_rewards.mean() < old_average_reward else "increased"} '
                        f'from {int(old_average_reward)} to {int(total_rewards.mean())}'
                    )
                    print(
                        f'Best reward {max(current_best_reward, int(total_rewards.mean()))}'
                    )

                if total_rewards.mean() > current_best_reward:
                    if save:
                        for net, name in [
                            [self.critic.main_net, 'critic_main'],
                            [self.critic.target_net, 'critic_target'],
                            [self.value, 'value'],
                            [self.actor, 'actor']
                        ]:
                            old_save_path = os.path.join(self.path, f'{name}_{int(current_best_reward)}.pt')
                            new_save_path = os.path.join(self.path, f'{name}_{int(total_rewards.mean())}.pt')

                            torch.save(net, new_save_path)
                            if os.path.isfile(old_save_path) and old_save_path != new_save_path:
                                os.remove(old_save_path)

    def _update_q(self, batch: list, gamma):
        '''
        Variable 'batch' should be a 1D list of Experiences - sorted or unsorted. The reward value should be the
        correct Q_s value for that state i.e., the cumulated discounted reward from that state onwards.
        '''

        # Calculate Q loss
        loss_qt = calc_iql_q_loss_batch(batch, self.device, self.critic, self.value, gamma)

        # Update Networks
        for network_name in ['critic_main', 'critic_target']:
            self.optimizer[network_name].zero_grad()
        loss_qt.backward()
        for network_name in ['critic_main', 'critic_target']:
            self.optimizer[network_name].step()

        return loss_qt.item()

    def _update_v(self, batch: list, tau):
        '''
        Variable 'batch' should be a 1D list of Experiences - sorted or unsorted. The reward value should be the
        correct Q_s value for that state i.e., the cumulated discounted reward from that state onwards.
        '''

        # Calculate V loss
        loss_v = calc_iql_v_loss_batch(batch, self.device, self.value, tau)

        # Update Networks
        self.optimizer['value'].zero_grad()
        loss_v.backward()
        self.optimizer['value'].step()

        return loss_v.item()

    def _update_policy(self, batch: list, beta=1, update_iter=4, ppo_clip=0.01):
        '''
        Variable 'batch' should be a 1D list of Experiences - sorted or unsorted. The reward value should be the
        correct Q_s value for that state i.e., the cumulated discounted reward from that state onwards.
        '''

        loss_policy = calc_iql_policy_loss_batch(batch, self.device, self.critic, self.value,
                                                 self.actor)

        self.optimizer['actor'].zero_grad()
        loss_policy.backward()
        self.optimizer['actor'].step()

        return loss_policy.item()

        '''
        # Unpack the batch
        states, actions = zip(*[(exp.state, exp.action) for exp in batch])

        # Calculate the logprobs of the action taken
        with torch.no_grad():
            old_action_logits = self.actor.forward(states, device=self.device)
            old_action_logprobs = torch.log_softmax(old_action_logits, dim=1)
            old_action_logprobs = old_action_logprobs.gather(1,
                                                             torch.tensor(actions).to(self.device).unsqueeze(
                                                                 -1)).squeeze(
                -1)
            old_action_logprobs = torch.where(torch.isinf(old_action_logprobs), -1000, old_action_logprobs)
            old_action_logprobs = torch.where(old_action_logprobs == 0, -1e-8, old_action_logprobs)

        total_loss_policy = 0

        for _ in range(update_iter):
            loss_policy = calc_iql_policy_loss_batch(batch, self.device, self.critic1, self.critic2,
                                                     self.value, self.actor, old_action_logprobs, beta,
                                                     ppo_clip)

            self.optimizer['actor'].zero_grad()
            loss_policy.backward()
            self.optimizer['actor'].step()

            total_loss_policy += loss_policy.item()

        return total_loss_policy
        '''

    def _update_behaviour_policy(self, batch: list):
        '''
        Variable 'batch' should be a 1D list of Experiences - sorted or unsorted. The reward value should be the
        correct Q_s value for that state i.e., the cumulated discounted reward from that state onwards.
        '''

        # Unpack the batch
        states, actions = zip(*[(exp.state, exp.action) for exp in batch])

        # Calculate the logprobs of the action taken
        logits = self.behaviour_policy.forward(states, device=self.device)
        logprobs = torch.log_softmax(logits, dim=1)
        logprobs = logprobs.gather(1, torch.tensor(actions).to(self.device).unsqueeze(-1)).squeeze(-1)

        # Calculate the loss and backprop
        loss = -logprobs.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, episodes=100, parallel_envs=32, verbose=True, behaviour=False):

        @ray.remote
        def env_run(actor):
            done = False
            prem_done = False
            state = self.env.reset()[0]
            ep_record = []
            while not done and not prem_done:
                with torch.no_grad():
                    logits = actor.forward(state, device='cpu')

                # action = torch.argmax(logits).item()
                action_probs = F.softmax(logits, dim=-1)
                action_distribution = Categorical(action_probs)

                action = action_distribution.sample().item()

                next_state, reward, done, prem_done, _ = self.env.step(action)

                ep_record.append(Experience(state=state,
                                            action=action,
                                            reward=reward,
                                            next_state=next_state))

                state = next_state

            total_reward = sum([exp.reward for exp in ep_record])

            return ep_record, total_reward

        @ray.remote
        def env_run_behaviour(actor):
            done = False
            prem_done = False
            state = self.env.reset()[0]
            ep_record = []
            while not done and not prem_done:
                with torch.no_grad():
                    logits = actor.forward(state, device='cpu')

                action_probs = F.softmax(logits, dim=-1)
                action_distribution = Categorical(action_probs)

                action = action_distribution.sample().item()

                next_state, reward, done, prem_done, _ = self.env.step(action)

                ep_record.append(Experience(state=state,
                                            action=action,
                                            reward=reward,
                                            next_state=next_state))

                state = next_state

            total_reward = sum([exp.reward for exp in ep_record])

            return ep_record, total_reward

        all_states = []
        all_actions = []
        all_rewards = []
        all_total_rewards = []
        all_next_states = []

        print(f'Beginning evaluation over {episodes} episodes...\n') if verbose else None
        for episode in tqdm(range(episodes // parallel_envs), disable=not verbose, file=sys.stdout):
            if behaviour:
                temp_batch_records, temp_total_reward = zip(
                    *ray.get([env_run_behaviour.remote(self.behaviour_policy.cpu()) for _ in range(parallel_envs)]))
            else:
                temp_batch_records, temp_total_reward = zip(
                    *ray.get([env_run.remote(self.actor.cpu()) for _ in range(parallel_envs)]))

            self.actor.to(self.device)

            temp_batch_states, temp_batch_actions, temp_batch_rewards, temp_batch_next_states = zip(
                *[(exp.state, exp.action, exp.reward, exp.next_state)
                  for episode in temp_batch_records for exp in episode])

            all_states += list(temp_batch_states)
            all_actions += list(temp_batch_actions)
            all_rewards += list(temp_batch_rewards)
            all_next_states += list(temp_batch_next_states)
            all_total_rewards += list(temp_total_reward)

        print(
            f'Evaluation complete! Average reward per episode: {np.array(all_total_rewards).mean()}') if verbose else None

        return np.array(all_states), np.array(all_actions), np.array(all_rewards), \
            np.array(all_next_states), np.array(all_total_rewards)

    def evaluate_offline(self, data, episodes=100, parallel_envs=32, verbose=True):

        dones = np.array([exp.done for exp in data])
        episode_idxs = np.where(dones)[0] + 1
        episode_idxs = episode_idxs.tolist()
        episode_idxs.insert(0, 0)
        episode_idxs = np.array(episode_idxs)
        episode_idxs = [slice(i, j) for i, j in zip(episode_idxs[:-1], episode_idxs[1:])]
        np.random.default_rng().shuffle(episode_idxs)

        for ep_idx in episode_idxs[:episodes]:
            p = 1.0
            h = []
            t = 0
            r = 0
            for exp in data[ep_idx]:
                h.append(exp.state)
                logits = self.actor.forward(exp.state, device='cpu')
                actionprobs = F.softmax(logits)[exp.action]
                p *= actionprobs

        # return np.array(all_states), np.array(all_actions), np.array(all_rewards), \
        # np.array(all_next_states), np.array(all_total_rewards)
        return

    def choose_action(self, state, device='cpu'):
        with torch.no_grad():
            logits = self.actor.forward(state, device=device)

        action = torch.argmax(logits)

        return action.item()

    def load_model(self, criticpath1=None, criticpath2=None, valuepath=None, actorpath=None, behaviourpolicypath=None):
        if criticpath1 is not None:
            if '_main' in criticpath1:
                self.critic1.main_net = torch.load(criticpath1)
                self.critic1.target_net = torch.load(criticpath1.replace('_main', '_target'))
            else:
                self.critic1.main_net = torch.load(criticpath1.replace('_target', '_main'))
                self.critic1.target_net = torch.load(criticpath1)
            self.critic1.has_net = True
            self.path = os.path.dirname(criticpath1)
        if criticpath2 is not None:
            if '_main' in criticpath1:
                self.critic2.main_net = torch.load(criticpath2)
                self.critic2.target_net = torch.load(criticpath2.replace('_main', '_target'))
            else:
                self.critic2.main_net = torch.load(criticpath2.replace('_target', '_main'))
                self.critic2.target_net = torch.load(criticpath2)
            self.critic2.has_net = True
        if valuepath is not None:
            self.value = torch.load(valuepath)
            self.value.has_net = True
        if actorpath is not None:
            self.actor = torch.load(actorpath)
            self.actor.has_net = True
        if behaviourpolicypath is not None:
            self.behaviour_policy = torch.load(behaviourpolicypath)
            self.behaviour_policy.has_net = True

        if self._compiled:
            self._compiled = False
            self._optimizer = None
            self._learning_rate = None
            self._regularisation = None
            print('Model loaded - recompile agent please')

'''
NEXT STEP - START CONSOLIDATING INCONSISTENCIES ACROSS CLASSES IN TERMS OF NET AND GET_ACTION
NOTE THAT ANIMATE CURRENTLY IS NOT CONSISTENT ACROSS ALL CLASSES
THEN - CONSIDER CREATING INDIVIDUAL ACTOR AND CRITIC NET CLASSES
'''
"""
