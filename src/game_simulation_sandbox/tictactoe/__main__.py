import anytree

from .game import TicTacToe


def main():
    g = TicTacToe()
    tree = anytree.Node(g.to_string(), g=g)
    leaves = [tree]
    print(0, len(leaves))
    for i in range(1, 11):
        cur_leaves = leaves
        leaves = []
        for leaf in cur_leaves:
            if not leaf.g.status():
                moves = leaf.g.make_moves()
                leaves += [
                    anytree.Node(move.to_string(), g=move, parent=leaf)
                    for move in moves
                ]
        print(i, len(leaves))


if __name__ == "__main__":
    main()
