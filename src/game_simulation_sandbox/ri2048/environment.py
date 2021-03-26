import abc
import sys

import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

from . import game

tf.compat.v1.enable_v2_behavior()

# Spec = tensor_spec.BoundedTensorSpec
Spec = array_spec.BoundedArraySpec


class Environment(py_environment.PyEnvironment):
    def __init__(self):
        super().__init__()
        self._action_spec = Spec(
            shape=(),
            dtype=np.int32,
            name="action",
            minimum=0,
            maximum=3,
        )
        self._observation_spec = Spec(
            shape=(16,),
            dtype=np.int32,
            name="observation",
            minimum=0,
            maximum=18,
        )
        self._reward_spec = Spec(
            shape=(),
            dtype=np.float32,
            name="reward",
            minimum=0.0,
            maximum=131072 / 2,
        )

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def reward_spec(self):
        return self._reward_spec

    def _reset(self) -> ts.TimeStep:
        self.game = game.Game()
        return ts.restart(self._represent())

    def _step(self, action) -> ts.TimeStep:
        # print(self.game.board)
        # print(game.directions[action].name)
        reward = np.array(
            self.game.move(game.directions[action]), dtype=np.float32
        )
        if self.game.is_game_over():
            return ts.termination(self._represent(), self.game.score)
        self.game.spawn()
        return ts.transition(self._represent(), reward, 1.0)

    def _represent(self):
        observation = np.zeros(16, dtype=np.int32)
        board_flat = self.game.board.flatten()
        for i in range(board_flat.shape[0]):
            if board_flat[i] > 0:
                exponent = int(np.round(np.log2(board_flat[i])))
                observation[i] = exponent + 1
        return observation


def validate():
    environment = Environment()
    utils.validate_py_environment(environment, episodes=5)


def make_tf_environment() -> tf_py_environment.TFPyEnvironment:
    return tf_py_environment.TFPyEnvironment(Environment())
