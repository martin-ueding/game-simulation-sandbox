import itertools
import pprint
from typing import *

import igraph

Piece = Tuple[int, int, int]


def main() -> None:
    pieces = make_all()
    pprint.pprint(pieces)
    print(len(pieces))
    g = make_graph(pieces)
    igraph.plot(g, layout=g.layout("sphere"), target="tridom.pdf")


def make_all() -> Set[Piece]:
    raw_pieces = itertools.product(range(6), range(6), range(6))
    unique_pieces = set(map(normalize_piece, raw_pieces))
    return unique_pieces


def normalize_piece(piece: Piece) -> Piece:
    a, b, c = piece
    unique_digits = len(set(piece))
    if unique_digits == 1:
        return piece
    elif unique_digits == 2:
        return tuple(sorted(piece))
    else:
        m = min(piece)
        if m == a:
            return a, b, c
        elif m == b:
            return b, c, a
        else:
            return c, a, b


def can_dock(left: Piece, right: Piece) -> bool:
    a, b, c = left
    x, y, z = right

    left_edges = [(a, b), (b, c), [c, a]]
    right_edges = [(x, y), (y, z), (z, x)]
    for left_edge in left_edges:
        for right_edge in right_edges:
            if left_edge == right_edge:
                return True
    return False


def make_graph(pieces: List[Piece]) -> igraph.Graph:
    pieces = list(pieces)
    g = igraph.Graph()
    for piece in pieces:
        g.add_vertex(
            name=str(piece), label=f"{piece[0]} {piece[1]} {piece[2]}", numbers=piece
        )
    for i, left in enumerate(pieces):
        for j, right in enumerate(pieces[i + 1 :]):
            if can_dock(left, right):
                g.add_edge(str(left), str(right))
    return g


if __name__ == "__main__":
    main()
