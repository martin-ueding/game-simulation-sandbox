from .transition import *


def test_padding() -> None:
    input = np.array([[1]])
    expected = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    padded = add_padding(input)
    assert np.all(padded == expected)


def test_remove_padding() -> None:
    input = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    expected = np.array([[1]])
    shrunk = remove_padding(input)
    assert np.all(shrunk == expected)


def test_next_state() -> None:
    initial = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    expected = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    next = next_state(initial)
    assert np.all(next == expected)
