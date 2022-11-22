from .agent import DQNAgent, PolicyGradient, ActorCritic


def create_agent(agent_type='doubleDQN'):
    agent_dict = {'doubleDQN': DQNAgent(),
                  'actorcritic': ActorCritic(),
                  'policy': PolicyGradient()}

    if agent_type not in agent_dict.keys():
        error = 'type must be one of: '
        for key in agent_dict.keys():
            error += key + ', '
        error = error[:-2]
        raise KeyError(error)

    return agent_dict[agent_type]