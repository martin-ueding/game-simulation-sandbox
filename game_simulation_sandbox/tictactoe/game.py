import copy
from typing import Generator
from typing import List

from ..treesearch.interface import BinaryValueFunction
from ..treesearch.interface import Observer
from ..treesearch.interface import TrajectoryCollector
from ..treesearch.interface import TreeIterator
from ..treesearch.interface import WinLossDraw
from ..treesearch.interface import WinLossDrawValue


class TicTacToe:
    def __init__(self, state=[" "] * 9, cross_player_turn=True):
        self.state = copy.copy(state)
        self.cross_player_turn = cross_player_turn

    def make_moves(self) -> List["TicTacToe"]:
        result = []
        for i, field in enumerate(self.state):
            if field == " ":
                new = TicTacToe(self.state, not self.cross_player_turn)
                new.state[i] = "x" if self.cross_player_turn else "o"
                new.state = normalize_state(new.state)
                if all(new.state != old.state for old in result):
                    result.append(new)
        return result

    def __str__(self) -> str:
        result = [
            "|".join(self.state[0:3]),
            "\n—————\n",
            "|".join(self.state[3:6]),
            "\n—————\n",
            "|".join(self.state[6:9]),
        ]
        return "".join(result)

    def to_string(self) -> str:
        return "|".join(
            [
                "".join(self.state[0:3]),
                "".join(self.state[3:6]),
                "".join(self.state[6:9]),
            ]
        )

    def status(self) -> str:
        for player in "xo":
            if set(self.state[0:3]) == {player}:
                return player
            if set(self.state[3:6]) == {player}:
                return player
            if set(self.state[6:9]) == {player}:
                return player
            if set(self.state[0::3]) == {player}:
                return player
            if set(self.state[1::3]) == {player}:
                return player
            if set(self.state[2::3]) == {player}:
                return player
            if set(self.state[0::4]) == {player}:
                return player
            if set(self.state[2:7:2]) == {player}:
                return player
        return ""


def normalize_state(state: List[str]) -> List[str]:
    rotate = [2, 5, 8, 1, 4, 7, 0, 3, 6]
    flip = [2, 1, 0, 5, 4, 3, 8, 7, 6]
    permutations = [state]
    for permutation_idx in range(3):
        permutations.append([permutations[-1][old_idx] for old_idx in rotate])
    for permutation in list(permutations):
        permutations.append([permutation[old_idx] for old_idx in flip])
    permutations.sort()
    return permutations[0]


class TicTacToeIterator(TreeIterator):
    def __init__(self, state: TicTacToe):
        self.state = state

    def get_children(self) -> Generator["TicTacToeIterator", None, None]:
        if self.state.status() != "":
            return
        for child in self.state.make_moves():
            yield TicTacToeIterator(child)


class PrintObserver(Observer):
    def observe(self, it: TreeIterator) -> None:
        assert isinstance(it, TicTacToeIterator)
        print(it.state)
        print(it.state.status())


class PrintTrajectoryCollector(TrajectoryCollector):
    def collect(self, trajectory: List[TreeIterator]) -> None:
        for it in trajectory:
            assert isinstance(it, TicTacToeIterator)
            print(it.state)
            print()


class TicTacToeValue(WinLossDrawValue):
    def get_outcome(self, it: TreeIterator) -> WinLossDraw:
        assert isinstance(it, TicTacToeIterator)
        status = it.state.status()
        if status == "x":
            return WinLossDraw.WIN
        elif status == "o":
            return WinLossDraw.LOSS
        else:
            return WinLossDraw.DRAW
