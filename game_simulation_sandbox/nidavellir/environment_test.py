from tf_agents.environments import utils

from game_simulation_sandbox.nidavellir.environment import Environment


def test_validate_environment():
    environment = Environment()
    utils.validate_py_environment(environment, episodes=5)
