import anytree.exporter

from .game import PrintObserver
from .game import PrintTrajectoryCollector
from .game import TicTacToe
from .game import TicTacToeIterator
from .game import TicTacToeValue
from game_simulation_sandbox.treesearch.mcts import AlternatingMonteCarloTreeSearch


def main():
    g = TicTacToe()
    observer = PrintObserver()
    WinLossDrawValue = TicTacToeValue()
    trajectory_collector = PrintTrajectoryCollector()
    tree_search = AlternatingMonteCarloTreeSearch(
        WinLossDrawValue, trajectory_collector, None
    )
    tree_search.run(TicTacToeIterator(g))

    dict_exporter = anytree.exporter.DictExporter(attriter=filter_attrs)
    json_exporter = anytree.exporter.JsonExporter(dict_exporter)

    tree_search.run_more(10000)

    for levelgroup in anytree.LevelOrderGroupIter(tree_search.tree):
        l = list(levelgroup)
        l.sort(key=lambda node: node.total)
        for node in l[:-3]:
            node.children = []

    graph_to_puml(tree_search.tree, "tictactoe-mcts.puml")


def filter_attrs(attributes):
    d = dict(attributes)
    return [
        ("wins", d["wins"]),
        ("total", d["total"]),
        ("ratio", None if d["total"] == 0 else d["wins"] / d["total"]),
        ("state", "[" + d["it"].state.to_string() + "]"),
    ]


header = """
@startmindmap

<style>
mindmapDiagram {
    node {
        BackgroundColor lightblue
    }
    :depth(2) {
      BackGroundColor lightcoral
    }
    :depth(4) {
      BackGroundColor lightcoral
    }
    :depth(6) {
      BackGroundColor lightcoral
    }
    :depth(8) {
      BackGroundColor lightcoral
    }
    :depth(10) {
      BackGroundColor lightcoral
    }
}
</style>
"""

footer = """
@endmindmap
"""


def graph_to_puml(root, filename) -> None:
    with open(filename, "w") as f:
        f.write(header)
        for pre, fill, node in anytree.RenderTree(root):
            it = node.it
            node_str = f"{node.wins}/{node.total} [{it.state.to_string()}]"
            f.write("*" * (node.depth + 1) + " " + node_str + "\n")
        f.write(footer)


if __name__ == "__main__":
    main()
