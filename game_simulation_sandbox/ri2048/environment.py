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

from . import game

tf.compat.v1.enable_v2_behavior()

# Spec = tensor_spec.BoundedTensorSpec
Spec = array_spec.BoundedArraySpec


class Environment(py_environment.PyEnvironment):
    def __init__(self, use_onehot=False):
        super().__init__()
        if use_onehot:
            self._represent = self._represent_onehot
            self._observation_spec = Spec(
                shape=(4, 4, 18),
                dtype=np.float32,
                name="observation",
                minimum=0.0,
                maximum=1.0,
            )
        else:
            self._represent = self._represent_dense
            self._observation_spec = Spec(
                shape=(16,),
                dtype=np.int32,
                name="observation",
                minimum=0,
                maximum=18,
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
        reward = np.array(self.game.move(game.directions[action]), dtype=np.float32)
        if self.game.is_game_over():
            return ts.termination(self._represent(), self.game.score)
        self.game.spawn()
        return ts.transition(self._represent(), reward, 1.0)

    def _represent_dense(self):
        observation = np.zeros(16, dtype=np.int32)
        board_flat = self.game.board.flatten()
        for i in range(board_flat.shape[0]):
            if board_flat[i] > 0:
                exponent = int(np.round(np.log2(board_flat[i])))
                observation[i] = exponent + 1
        return observation

    def _represent_onehot(self):
        observation = np.zeros((4, 4, 18), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                if self.game.board[i, j] > 0:
                    exponent = int(np.round(np.log2(self.game.board[i, j])))
                    observation[i, j, exponent] = 1.0
        return observation


def validate():
    environment = Environment()
    utils.validate_py_environment(environment, episodes=5)


def make_tf_environment() -> tf_py_environment.TFPyEnvironment:
    return tf_py_environment.TFPyEnvironment(Environment(True))
