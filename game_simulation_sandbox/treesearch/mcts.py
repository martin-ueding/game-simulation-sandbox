import anytree

from .interface import BinaryValueFunction
from .interface import Observer
from .interface import TrajectoryCollector
from .interface import TreeIterator
from .interface import TreeSearch


class MonteCarloTreeSearch(TreeSearch):
    def __init__(
        self,
        binary_value_function: BinaryValueFunction,
        trajectory_collector: TrajectoryCollector,
        observer: Observer,
    ):
        self.binary_value_function = binary_value_function
        self.trajectory_collector = trajectory_collector
        self.observer = observer

    def run(self, start: TreeIterator) -> None:
        tree = anytree.Node(start, wins=0, total=0)
