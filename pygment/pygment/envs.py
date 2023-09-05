from typing import Dict, Any, List

import gymnasium
import numpy as np


class VariableTimeSteps(gymnasium.Wrapper):
    """
    Custom wrapper to implement a variable time-step environment.
    """

    def __init__(self, env, fn=None, max_time_steps=10):
        """
        :param env: Gymnasium environment
        :param fn: Custom function to generate a non-random time-delay - it is assumed that this function already
        includes a maximum time-delay.
        :param max_time_steps: If no custom function provided, a random time-delay is generated between 0
        and max_time_steps.
        """
        super().__init__(env)
        self.fn = fn
        self.max_time_steps = max_time_steps

    def step(self, action):
        state: np.ndarray
        reward: float
        done: bool
        prem_done: bool
        info: Dict[str, Any]
        sequential_reward: List[float] = []

        state, reward, done, prem_done, info = self.step(action)
        sequential_reward += [reward]

        if done or prem_done:
            info['time_steps'] = 1
            return state, sequential_reward, done, prem_done, info

        time_steps = self._generate_time_steps(state)

        for step in range(2, time_steps + 2):
            state, reward, done, prem_done, info = self.step(action)
            sequential_reward += [reward]

            if done or prem_done:
                info['time_steps'] = step
                return state, sequential_reward, done, prem_done, info

        info['time_steps'] = time_steps + 1
        return state, sequential_reward, done, prem_done, info

    def _generate_time_steps(self, state):
        if self.fn is None:
            return np.random.randint(0, self.max_time_steps + 1)

        else:
            return self.fn(state)

