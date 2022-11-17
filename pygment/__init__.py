from agent import DQNAgent


def create_agent(type='doubleDQN'):
    if type not in ['doubleDQN',
                    'actorcritic']:
        raise KeyError('type must be one of: "doubleDQN", "actorcritic"')

    # Eventually, will need an if-else statement related to type above

    return DQNAgent()