import anytree

from .game import PrintObserver
from .game import PrintTrajectoryCollector
from .game import TicTacToe
from .game import TicTacToeIterator
from .game import TicTacToeWin
from game_simulation_sandbox.treesearch.mcts import MonteCarloTreeSearch


def main():
    g = TicTacToe()
    observer = PrintObserver()
    binary_value_function = TicTacToeWin()
    trajectory_collector = PrintTrajectoryCollector()
    tree_search = MonteCarloTreeSearch(
        binary_value_function, trajectory_collector, observer
    )
    tree_search.run(TicTacToeIterator(g))


if __name__ == "__main__":
    main()
