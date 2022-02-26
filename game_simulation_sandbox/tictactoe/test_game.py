from .game import normalize_state
from .game import TicTacToe


def test_empty() -> None:
    g = TicTacToe()
    assert g.to_string() == "   |   |   "


def test_normalization() -> None:
    assert normalize_state(list(" x       ")) == list("       x ")
    assert normalize_state(list("ox       ")) == list("       xo")
