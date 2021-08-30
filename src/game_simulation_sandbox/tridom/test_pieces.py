from .pieces import *


def test_dock() -> None:
    assert can_dock((0, 0, 1), (0, 4, 1))
