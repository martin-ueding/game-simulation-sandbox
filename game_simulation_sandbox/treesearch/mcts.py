import math
import random
from typing import Optional
from typing import Union

import anytree
import numpy as np

from .interface import Observer
from .interface import TrajectoryCollector
from .interface import TreeIterator
from .interface import TreeSearch


class MCTSBackpropagation:
    def get_win_score(
        self, node: TreeIterator, start_players_turn: bool
    ) -> Union[int, float]:
        raise NotImplementedError


class MonteCarloTreeSearch(TreeSearch):
    def __init__(
        self,
        backpropagation: MCTSBackpropagation,
        trajectory_collector: TrajectoryCollector,
        observer: Optional[Observer],
    ):
        self.backpropagation = backpropagation
        self.trajectory_collector = trajectory_collector
        self.observer = observer
        self.tree: Optional[anytree.AnyNode] = None

    def run(self, start: TreeIterator) -> None:
        self.tree = anytree.AnyNode(it=start, wins=0, total=0)

    def run_more(self, iterations: int) -> None:
        for i in range(iterations):
            # Selection phase.
            node = self.tree
            start_players_turn = False
            if self.observer is not None:
                self.observer.observe(node.it)
            while len(node.children) > 0:
                if min(child.total for child in node.children) == 0:
                    node = random.choice(
                        [child for child in node.children if child.total == 0]
                    )
                else:
                    child_values = [
                        child.wins / child.total
                        + math.sqrt(2 * math.log(node.total) / child.total)
                        for child in node.children
                    ]
                    top_child = node.children[np.argmax(child_values)]
                    node = top_child
                start_players_turn = not start_players_turn
                if self.observer is not None:
                    self.observer.observe(node.it)

            # Expansion and simulation phase.
            while True:
                # Create new children, if that is possible.
                for child_it in node.it.get_children():
                    anytree.AnyNode(parent=node, it=child_it, wins=0, total=0)

                # If there are no children, we are on a game-ending state.
                if len(node.children) == 0:
                    break

                # Otherwise we just pick one of the children at random.
                node = random.choice(node.children)
                start_players_turn = not start_players_turn
                if self.observer is not None:
                    self.observer.observe(node.it)

            # Backpropagation phase.
            for x in node.iter_path_reverse():
                x.total += 1
                x.wins += self.backpropagation.get_win_score(
                    node.it, start_players_turn
                )
                start_players_turn = not start_players_turn
