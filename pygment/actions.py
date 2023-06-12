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
                                     min=1 - epsilon,
                                     max=1 + epsilon) * advantage
    policy_loss = torch.min(ratio_loss, clipped_ratio_loss)

    policy_loss = -policy_loss.mean()

    beta = 0.01
    entropy_loss = beta * batch_entropy if batch_entropy else 0

    return policy_loss, value_loss, entropy_loss


def calc_iql_v_loss_batch(batch, device, actor1, actor2, critic, tau):
    # Unpack the batch
    states, actions, reward, dones = zip(*[(exp.state, exp.action, exp.reward, exp.done) for exp in batch])

    # Calculate Q(s',a), Q'(s',a), and V(s) for each state in the batch
    with torch.no_grad():
        pred_Q1 = actor1.forward(states, target=False, device=device)
        pred_Q2 = actor2.forward(states, target=False, device=device)
        pred_Q = torch.minimum(pred_Q1, pred_Q2).gather(1, torch.tensor(actions).to(device).unsqueeze(-1)).squeeze(-1)

    pred_V_s = critic.forward(states, device=device).squeeze(-1)

    # Calculate loss_v
    loss_v = pred_V_s - pred_Q
    mask = loss_v > 0
    loss_v = loss_v ** 2
    loss_v[mask] = loss_v[mask] * (1 - tau)
    loss_v[~mask] = loss_v[~mask] * tau

    return loss_v.mean()


def calc_iql_q_loss_batch(batch, device, critic1, critic2, value, gamma):
    # Unpack the batch
    states, actions, reward, next_states, next_actions, cum_rewards, dones = zip(*[(exp.state, exp.action, exp.reward,
                                                                                    exp.next_state, exp.next_action,
                                                                                    exp.cum_reward, exp.done)
                                                                                   for exp in batch])

    # Calculate Q(s,a) for each state in the batch - V(s) is updated from this
    pred_Q1 = critic1.forward(states, target=False, device=device)
    pred_Q1_choice = pred_Q1.gather(1, torch.tensor(actions).to(device).unsqueeze(-1)).squeeze(-1)
    pred_Q2 = critic2.forward(states, target=False, device=device)
    pred_Q2_choice = pred_Q2.gather(1, torch.tensor(actions).to(device).unsqueeze(-1)).squeeze(-1)

    # Calculate Q_t(s,a) for each state in the batch - this is the 'optimal' Q-function, updated from V(s')
    pred_Q1_t = critic1.forward(states, target=True, device=device)
    pred_Q1_t_choice = pred_Q1_t.gather(1, torch.tensor(actions).to(device).unsqueeze(-1)).squeeze(-1)
    pred_Q2_t = critic2.forward(states, target=True, device=device)
    pred_Q2_t_choice = pred_Q2_t.gather(1, torch.tensor(actions).to(device).unsqueeze(-1)).squeeze(-1)

    # Calculate V(s') for each state in the batch
    with torch.no_grad():
        pred_V_s_next = value.forward(next_states, device=device).squeeze(-1)
        pred_V_s_next = torch.where(~torch.tensor(dones).to(device), pred_V_s_next, torch.zeros_like(pred_V_s_next))

    """
    
    # Calculate Q'(s',a') for the next state in the batch
    with torch.no_grad():
        pred_Q1_next = critic1.forward(next_states, target=True, device=device)
        pred_Q1_next_choice = pred_Q1_next.gather(1, torch.tensor(next_actions).to(device).unsqueeze(-1)).squeeze(-1)
        pred_Q2_next = critic2.forward(next_states, target=True, device=device)
        pred_Q2_next_choice = pred_Q2_next.gather(1, torch.tensor(next_actions).to(device).unsqueeze(-1)).squeeze(-1)
        pred_Q_next_choice = torch.min(pred_Q1_next_choice, pred_Q2_next_choice)
        pred_Q_next_choice = torch.where(~torch.tensor(dones).to(device), pred_Q_next_choice,
                                         torch.zeros_like(pred_Q_next_choice))
    """
    # Calculate loss_q
    # target_q = torch.tensor(reward, dtype=torch.float32).to(device) + gamma * pred_V_s
    # target_q = torch.tensor(reward, dtype=torch.float32).to(device) + gamma * pred_Q_next_choice
    target_q = torch.tensor(cum_rewards, dtype=torch.float32).to(device)

    loss_q1 = F.mse_loss(pred_Q1_choice, target_q)
    loss_q2 = F.mse_loss(pred_Q2_choice, target_q)

    # Calculate loss_q1
    target_q = torch.tensor(reward, dtype=torch.float32).to(device) + gamma * pred_V_s_next
    loss_qt1 = F.mse_loss(pred_Q1_t_choice, target_q)
    loss_qt2 = F.mse_loss(pred_Q2_t_choice, target_q)

    return loss_q1, loss_q2, loss_qt1, loss_qt2


def calc_iql_policy_loss_batch(batch, device, critic1, critic2, value, actor, old_action_logprobs, beta,
                               ppo_clip):
    # Unpack the batch
    states, actions = zip(*[(exp.state, exp.action) for exp in batch])

    # Calculate Qt(s,a) for each state in the batch
    with torch.no_grad():
        pred_Q1 = critic1.forward(states, target=True, device=device)
        pred_Q2 = critic2.forward(states, target=True, device=device)
        pred_Q = torch.minimum(pred_Q1, pred_Q2)
        pred_Q = pred_Q.gather(1, torch.tensor(actions).to(device).unsqueeze(-1)).squeeze(-1)

    # Calculate the logprobs of the action taken
    action_logits = actor.forward(states, device=device)
    action_logprobs = torch.log_softmax(action_logits, dim=1)
    action_logprobs = action_logprobs.gather(1, torch.tensor(actions).to(device).unsqueeze(-1)).squeeze(-1)
    action_logprobs = torch.where(torch.isinf(action_logprobs), -1000, action_logprobs)
    action_logprobs = torch.where(action_logprobs == 0, -1e-8, action_logprobs)

    # Calculate V(s) for each state in the batch
    with torch.no_grad():
        pred_V_s = value.forward(states, device=device).squeeze(1)

    # Calculate Advantage
    advantage = pred_Q - pred_V_s
    advantage = (advantage - advantage.mean()) / (torch.max(advantage.std() + 1e-8))
    advantage = torch.exp(beta * advantage)

    # Calculate the policy loss
    ratio = torch.exp(action_logprobs - old_action_logprobs) - 1
    clipped_ratio_pos_adv = torch.clamp(ratio, min=-ppo_clip)
    clipped_ratio_neg_adv = torch.clamp(ratio, max=ppo_clip)
    """
    The goal of the PPO-like loss function is the following:
    - We always want to trend towards a positive ratio, because that means we are moving in the direction of the
    behavioural policy.
    - So when the ratio is positive, the loss (which gets inverted) should be positive.
    - And when the ratio is negative, the loss (which gets inverted) should be negative.
    
    - We prefer to make strong moves when the advantage is positive (i.e., >1) and the ratio is positive.
    This is expressed as a large positive loss, which is good.
    
    - We prefer to make strong moves when the advantage is negative (i.e., <1) and the ratio is negative.
    This is expressed as a small negative loss (because the bigger the negative advantage, the closer the value is to 0).
    
    - We prefer to make weak moves when the advantage is positive (i.e., >1) and the ratio is negative.
    If we clip the ratio to avoid large negative ratios, then the overall loss will be negative but small.
    
    - We prefer to make weak moves when the advantage is negative (i.e., <1) and the ratio is positive.
    If we clip the ratio to avoid large positive ratios, then the overall loss will be positive but small.
    
    So, in summary:
    1. Positive ratio, positive advantage -> positive loss (which gets inverted i.e., is good)
    2. Negative ratio, positive advantage -> negative loss, clipped and small (which gets inverted i.e., is bad)
    
    3. Positive ratio, negative advantage -> positive loss, but clipped and small (which gets inverted i.e., is good)
    4. Negative ratio, negative advantage -> negative loss (which gets inverted i.e., is bad)
    """
    loss = torch.where(advantage > 1,
                       # When advantage is "positive"...
                       # If ratio is positive, then loss is positive (good) i.e., ratio * advantage.
                       # If ratio is negative, clip at -0.2, and loss is negative (bad) i.e., clipped(ratio) * advantage
                       torch.max(ratio, clipped_ratio_pos_adv) * advantage,
                       # When advantage is "negative"...
                       # If ratio is positive, then loss is positive (bad) but clipped i.e., clipped(ratio) * advantage
                       # If ratio is negative, then loss is negative (good) i.e., ratio * advantage
                       torch.min(ratio, clipped_ratio_neg_adv) * advantage)

    # loss = torch.exp(beta * (pred_Q - pred_V_s)) * action_logprobs
    loss = -loss.mean()

    return loss, action_logprobs
