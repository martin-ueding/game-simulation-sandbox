import anytree

from ..treesearch.random_walk import RandomWalk
from .game import PrintObserver
from .game import TicTacToe
from .game import TicTacToeIterator


def main():
    g = TicTacToe()
    observer = PrintObserver()
    tree_search = RandomWalk(observer)
    tree_search.run(TicTacToeIterator(g))


if __name__ == "__main__":
    main()
