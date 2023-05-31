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


def calc_entropy_loss_policy(action_probs, action_logprobs, entropy_beta=0.01, device='cpu'):
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


def calc_loss_actor_critic(batch_Q_s, batch_actions, batch_entropy, batch_action_logprobs,
                           batch_state_values, device='cpu',
                           epsilon=0.2, batch_old_policy_logprobs=None, advantage=None):

    # start with value gradients
    value_loss = F.mse_loss(batch_state_values, batch_Q_s)

    # and then the actor gradients
    if advantage is None:
        advantage = batch_Q_s - batch_state_values.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)


    policy_ratio = torch.exp(batch_action_logprobs - batch_old_policy_logprobs).reshape(-1, batch_actions.shape[-1])

    ratio_loss = policy_ratio * advantage
    '''clipped_ratio_loss = torch.where(advantage > 0,
                                     (1 + epsilon) * advantage,
                                     (1 - epsilon) * advantage)'''
    clipped_ratio_loss = torch.clamp(policy_ratio,
                                     min=1-epsilon,
                                     max=1+epsilon) * advantage
    policy_loss = torch.min(ratio_loss, clipped_ratio_loss)

    policy_loss = -policy_loss.mean()

    beta = 0.01
    entropy_loss = beta * batch_entropy if batch_entropy else 0

    return policy_loss, value_loss, entropy_loss


def calc_iql_loss_batch(batch, device, model, gamma, tau):
    # Unpack the batch
    states, actions, reward, cum_reward, next_states = zip(*[(exp.state, exp.action, exp.reward, exp.cum_reward,
                                                              exp.next_state) for exp in batch])

    # Calculate Q(s,a) and V(s) for each state in the batch
    pred_Q_s, pred_V_s = model.forward(states, device=device)
    pred_Q_s = pred_Q_s.gather(1, torch.tensor(actions).to(device).unsqueeze(-1)).squeeze(-1)
    pred_V_s = pred_V_s.squeeze(-1)

    # Calculate loss for V(s)
    loss_v = pred_V_s - torch.tensor(cum_reward, dtype=torch.float32).to(device)
    mask = loss_v > 0
    loss_v = loss_v ** 2
    loss_v[mask] = loss_v[mask] * (1 - tau)
    loss_v[~mask] = loss_v[~mask] * tau

    # Calculate loss for Q(s,a) using the Value function
    loss_q = torch.tensor(reward, dtype=torch.float32).to(device) + gamma * pred_V_s - pred_Q_s
    loss_q = loss_q ** 2

    return loss_q.mean(), loss_v.mean()


def calc_iql_policy_loss_batch(batch, device, model, beta):
    # Unpack the batch
    states, actions, reward, cum_reward, next_states = zip(*[(exp.state, exp.action, exp.reward, exp.cum_reward,
                                                              exp.next_state) for exp in batch])

    # Calculate Q(s,a) and V(s) for each state in the batch
    pred_Q_s, pred_V_s = model.forward(states, device=device)
    # Set Q_s logits as normalised probabilities, and convert back to log
    pred_Q_s = torch.softmax(pred_Q_s, dim=1)
    pred_Q_s = pred_Q_s.gather(1, torch.tensor(actions).to(device).unsqueeze(-1)).squeeze(-1)
    pred_Q_s = torch.log(pred_Q_s)
    # Address any -inf values
    pred_Q_s = torch.where(torch.isinf(pred_Q_s), -1000, pred_Q_s)
    # Address any zero values, so that the advantage part is still retained
    pred_Q_s = torch.where(pred_Q_s == 0, -1e-8, pred_Q_s)
    # Detach the value network, which should remain unchanged
    pred_V_s = pred_V_s.detach().squeeze(1)

    # Calculate the policy loss
    loss = torch.exp(beta * (torch.tensor(cum_reward, dtype=torch.float32).to(device) - pred_V_s)) * pred_Q_s
    loss = -loss.mean()

    return loss


