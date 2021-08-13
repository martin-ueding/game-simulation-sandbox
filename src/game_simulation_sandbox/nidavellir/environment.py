import abc
import sys

import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.environments import wrappers
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts

from . import coin_upgrades

tf.compat.v1.enable_v2_behavior()

# Spec = tensor_spec.BoundedTensorSpec
Spec = array_spec.BoundedArraySpec


class Environment(py_environment.PyEnvironment):
    def __init__(self):
        super().__init__()
        self._observation_spec = Spec(
            shape=(4,),
            dtype=np.int32,
            name="observation",
            minimum=2,
            maximum=25,
        )
        self._action_spec = Spec(
            shape=(),
            dtype=np.int32,
            name="action",
            minimum=0,
            maximum=3,
        )
        self._reward_spec = Spec(
            shape=(),
            dtype=np.float32,
            name="reward",
            minimum=0.0,
            maximum=58.0,
        )
        self._reset()

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def reward_spec(self):
        return self._reward_spec

    def _reset(self) -> ts.TimeStep:
        self.coins = [2, 3, 4, 5]
        self.bank = coin_upgrades.make_bank(self.coins)
        self.round = 0
        return ts.restart(self._represent())

    def _step(self, action) -> ts.TimeStep:
        selected = self.coins[action]
        keep = min(selected)
        new = min(selected) + max(selected)
        while new <= 25 and self.bank[new] == 0:
            new += 1
        if new <= 25:
            self.coins.remove(max(selected))
            self.coins.append(new)
        self.round += 1
        if self.round == 8:
            return ts.termination(self._represent(), sum(self.coins))
        return ts.transition(self._represent(), 0, 1.0)

    def _represent(self):
        return np.array(self.coins)


def validate():
    environment = Environment()
    utils.validate_py_environment(environment, episodes=5)


def make_tf_environment() -> tf_py_environment.TFPyEnvironment:
    return tf_py_environment.TFPyEnvironment(Environment(True))
