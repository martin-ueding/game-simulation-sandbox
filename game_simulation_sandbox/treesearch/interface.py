import enum
from typing import Generator
from typing import List


class TreeIterator:
    def get_children(self) -> Generator["TreeIterator", None, None]:
        raise NotImplementedError()


class TreeSearch:
    def run(self, start: TreeIterator) -> None:
        raise NotImplementedError()


class Observer:
    def observe(self, state: TreeIterator) -> None:
        raise NotImplementedError()


class TrajectoryCollector:
    def collect(self, trajectory: List[TreeIterator]) -> None:
        raise NotImplementedError()


class BinaryValueFunction:
    def is_success(self, state: TreeIterator) -> bool:
        raise NotImplementedError()


class WinLossDraw(enum.Enum):
    WIN = 1
    LOSS = 2
    DRAW = 3


class WinLossDrawValue:
    def get_outcome(self, state: TreeIterator) -> WinLossDraw:
        raise NotImplementedError()
