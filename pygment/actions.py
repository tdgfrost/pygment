import numpy as np
import torch
import torch.nn.functional as F


def GreedyEpsilonSelector(obs, epsilon, net):
    if np.random.random() <= epsilon:
        action = np.random.randint(net.action_space)

    else:
        with torch.no_grad():
            q_values = net.forward(obs, target=False)
            action = torch.argmax(q_values).item()

    return action


def unpack_batch(batch: list):
    states, actions, rewards, next_states, dones = zip(*[(exp.state, exp.action, exp.reward, exp.next_state, exp.done)
                                                         for exp in batch])

    return states, actions, rewards, next_states, dones


def calc_loss_batch(batch, device, model, gamma):
    # Function for returning both the losses and the prioritised samples
    states, actions, rewards, next_states, dones = unpack_batch(batch)

    states_v, actions_v, rewards_v, done_mask = torch.tensor(states).to(device), torch.tensor(actions).to(device), \
                                                torch.tensor(rewards, dtype=torch.float32).to(device), torch.tensor(dones, dtype=torch.bool).to(device)

    state_action_values = model.forward(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_states_v = torch.tensor(next_states).to(device)
        next_state_values = model.forward(next_states_v, target=True).max(1)[0]
        next_state_values[done_mask] = 0.0
        expected_state_action_values = next_state_values.detach() * gamma + rewards_v

    loss_v = F.mse_loss(state_action_values, expected_state_action_values)
    return loss_v


def calc_loss_policy(cum_rewards, actions, action_logprobs, device):
    cum_rewards = torch.tensor(cum_rewards, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).to(device)
    if type(action_logprobs) is list:
        action_logprobs = torch.stack(action_logprobs).to(device)
    action_logprobs = action_logprobs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    loss = cum_rewards * action_logprobs
    loss = -loss.mean()

    return loss


def calc_entropy_loss_policy(action_probs, action_logprobs, entropy_beta=0.01, device='mps'):
  if type(action_probs) is list:
      action_probs = torch.stack(action_probs).to(device)
  if type(action_logprobs) is list:
      action_logprobs = torch.stack(action_logprobs).to(device)

  # This is necessary to avoid -inf loss (when p*log(p) is 0 * -inf)
  action_logprobs = torch.where(torch.isinf(action_logprobs),
                                -1000,
                                action_logprobs)

  entropy_loss = -(action_probs * action_logprobs).sum(1).mean()
  entropy_loss *= -entropy_beta

  return entropy_loss


def calc_cum_rewards(rewards_record, gamma):
    cum_rewards = []
    cum_r = 0.0
    for r in reversed(rewards_record):
        cum_r *= gamma
        cum_r += r
        cum_rewards.append(cum_r)

    return cum_rewards[::-1]


def calc_loss_actor_critic(batch_Q_s, batch_actions, batch_action_probs, batch_action_logprobs,
                           batch_state_values, device='mps'):

    # start with value gradients
    value_loss = F.mse_loss(batch_state_values, batch_Q_s)

    # and then the actor gradients
    advantage = batch_Q_s - batch_state_values.detach()

    policy_loss = batch_action_logprobs.gather(1, batch_actions)
    policy_loss *= advantage
    policy_loss = -policy_loss.mean()

    beta = 0.01
    entropy_loss = beta * (batch_action_probs * batch_action_logprobs).sum(1).mean()

    #loss = policy_loss + 0.5 * value_loss - entropy_loss
    loss = policy_loss + value_loss - entropy_loss

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

