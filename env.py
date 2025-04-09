import gymnasium
import numpy as np

from functools import lru_cache
from physical_quoridor import PhysicalQuoridorEnv


def convert_to_box(value, min_value, max_value):
    return value * (max_value - min_value) + min_value


def convert_to_discrete(value, n):
    return int(convert_to_box(max(value - 1e-9, 0), 0, n))


def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)


class PhysicalQuoridorEnv_(PhysicalQuoridorEnv):
    @classmethod
    def convert_actions(cls, actions):
        return dict(zip(
            actions.keys(),
            map(
                lambda action: (
                    convert_to_discrete(action[0], 2),
                    [
                        convert_to_box(action[1], -1, 1),
                        convert_to_box(action[2], -1, 1)
                    ],
                    (
                        convert_to_discrete(action[3], 8),
                        convert_to_discrete(action[4], 8),
                        convert_to_discrete(action[5], 2),
                    )
                ),
                actions.values()
            )
        ))

    @classmethod
    def convert_observations(cls, observations):
        return dict(zip(
            observations.keys(),
            map(
                lambda observation: np.array(
                    [
                        normalize(observation[0][0], -5, 5),
                        normalize(observation[0][1], -5, 5),
                        normalize(observation[1][0], -20, 20),
                        normalize(observation[1][1], -20, 20),

                        normalize(observation[2][0], -5, 5),
                        normalize(observation[2][1], -5, 5),
                        normalize(observation[3][0], -20, 20),
                        normalize(observation[3][1], -20, 20),

                        *np.ravel(observation[4]).astype(np.float32),

                        normalize(observation[5], 0, 10),
                        normalize(observation[6], 0, 10)
                    ],
                    dtype=np.float32
                ),
                observations.values()
            )
        ))

    def reset(self, seed=None, options=None):
        observations, infos = super().reset(seed=seed, options=options)

        return self.convert_observations(observations), infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = super().step(self.convert_actions(actions))

        return self.convert_observations(observations), rewards, terminations, truncations, infos

    @lru_cache(maxsize=None)
    def action_space(self, agent):
        return gymnasium.spaces.Box(0, 1, shape=[6], dtype=np.float32)

    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        return gymnasium.spaces.Box(0, 1, [2 + 2 + 2 + 2 + 8 * 8 * 2 + 1 + 1], dtype=np.float32)
