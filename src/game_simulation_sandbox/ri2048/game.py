import collections
import random
import typing

import numpy as np


Direction = collections.namedtuple("Direction", ["name", "transpose", "reverse"])

directions = [
    Direction("down", True, True),
    Direction("left", False, False),
    Direction("right", False, True),
    Direction("up", True, False),
]


class Game(object):
    def __init__(self):
        self.board = np.zeros((4, 4), dtype="int")
        self.score = 0
        self.steps = 0
        self.spawn()
        self.spawn()

    def spawn(self) -> None:
        new_number = random.choices([2, 4], [0.9, 0.1])[0]
        free_cols, free_rows = self.get_free_fields()
        new_coord = random.choice(list(zip(free_cols, free_rows)))
        self.board[new_coord] = new_number

    def get_free_fields(self) -> typing.Tuple[np.array, np.array]:
        free_cols, free_rows = np.where(self.board == 0)
        return free_cols, free_rows

    def is_game_over(self) -> bool:
        return not np.any(self.board == 0)

    def move(self, direction: Direction) -> int:
        sum_merges = 0
        board = transform_board(self.board, direction, True)
        for i in range(board.shape[0]):
            row = board[i]
            non_zero = list(row[row != 0])
            j = 0
            while j < len(non_zero) - 1:
                if non_zero[j] == non_zero[j + 1]:
                    non_zero[j] += non_zero[j + 1]
                    sum_merges += non_zero[j]
                    del non_zero[j + 1]
                j += 1
            row = non_zero + [0] * (4 - len(non_zero))
            board[i, :] = row
        self.board = transform_board(board, direction, False)
        self.score += sum_merges
        self.steps += 1
        return sum_merges

    def __str__(self) -> str:
        return str(self.board)


def transform_board(board: np.array, direction: Direction, forward: bool) -> np.array:
    if forward:
        if direction.transpose:
            board = board.T
        if direction.reverse:
            board = board[:, ::-1]
    else:
        if direction.reverse:
            board = board[:, ::-1]
        if direction.transpose:
            board = board.T
    return board
