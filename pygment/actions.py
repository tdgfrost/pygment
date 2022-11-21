import numpy as np
import torch
import torch.nn.functional as F


def GreedyEpsilonSelector(obs, epsilon, net):
    action_space = net[-1].out_features
    if np.random.random() <= epsilon:
        action = np.random.randint(action_space)

    else:
        with torch.no_grad():
            q_values = net(obs)
            action = torch.argmax(q_values).cpu().detach().item()
            #action_p = F.softmax(q_values, dim=0).detach().cpu().numpy()
            #action = np.random.choice([i for i in range(action_space)],
                                      #p=action_p)

    return action


def unpack_batch(batch: list):
    states, actions, rewards, next_states, dones = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.done)
        #if exp.next_state is None:
            #next_states.append(exp.)  # the result will be masked anyway
        #else:
            #lstate = np.array(exp.last_state)
        next_states.append(exp.next_state)
    return np.array(states, copy=False), np.array(actions), \
           np.array(rewards, dtype=np.float32), \
           np.array(next_states, copy=False), \
           np.array(dones, dtype=np.uint8)


def calc_loss_batch(batch, device, model, gamma):
    # Function for returning both the losses and the prioritised samples
    states, actions, rewards, next_states, dones = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.tensor(dones, dtype=torch.bool).to(device)

    state_action_values = model.main_net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_states_v = torch.tensor(next_states).to(device)
        next_state_values = model.target_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        expected_state_action_values = next_state_values.detach() * gamma + rewards_v

    loss_v = F.mse_loss(state_action_values, expected_state_action_values)
    return loss_v


def calc_loss_policy(cum_rewards, actions, action_logprobs, action_space, device):
    cum_rewards = torch.tensor(cum_rewards, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).to(device)
    action_logprobs = torch.tensor(action_logprobs, dtype=torch.float32).to(device)
    action_space = torch.tensor(action_space, dtype=torch.int).to(device)

    loss = cum_rewards * action_logprobs[:, actions]
    loss = loss.mean()

    return loss


def calc_cum_rewards(rewards_record, gamma):
    cum_rewards = []
    cum_r = 0.0
    for r in reversed(rewards_record):
        cum_r *= gamma
        cum_r += r
        cum_rewards.append(cum_r)

    return cum_rewards[::-1]


def calc_loss_actor_critic(rewards, logprobs, state_values, gamma=0.99):

    # rewards as input should be self.rewards for the agent
    # logprobs should be self.logprobs
    # state_values should be self.state_values
    # gamma should be self.gamma -> maybe won't need a default if self.gamma has a default value?

    disc_rewards = []
    disc_reward = 0
    for reward in rewards:
        disc_reward = reward + gamma * disc_reward
        rewards.insert(0, disc_reward)


    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std())

    loss = 0
    for logprob, value, reward in zip(logprobs, state_values, rewards):
        advantage = reward - value.item()
        action_loss = -logprob * advantage
        value_loss = F.smooth_l1_loss(value, reward)
        loss += action_loss + value_loss

    return loss


def calc_loss_prios(batch, batch_weights, device, model, gamma):
    # Function for returning both the losses and the prioritised samples
    states, actions, rewards, next_states, dones = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.tensor(dones, dtype=torch.bool).to(device)
    batch_weights_v = torch.tensor(batch_weights, dtype=torch.float32).to(device)

    state_action_values = model.main_net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_states_v = torch.tensor(next_states).to(device)
        next_state_values = model.target_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        expected_state_action_values = next_state_values.detach() * gamma + rewards_v

    batch_weights = batch_weights ** 0.6
    batch_weights = batch_weights / batch_weights.sum()

    losses_v = batch_weights_v * (state_action_values - expected_state_action_values) ** 2
    return losses_v.mean(), (losses_v + 1e-5).data.cpu().numpy()

