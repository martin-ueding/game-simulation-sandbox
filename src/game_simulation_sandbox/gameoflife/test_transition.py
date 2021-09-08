from .transition import *


def test_padding() -> None:
    input = np.array([[1]])
    expected = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    padded = add_padding(input)
    assert np.all(padded == expected)
