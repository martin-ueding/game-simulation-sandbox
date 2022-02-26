import itertools
import pathlib
import pprint
from typing import *

import igraph

from game_simulation_sandbox.igraph_util import render_igraph_neato

Piece = Tuple[int, int, int]


def main() -> None:
    pieces = make_all()
    pprint.pprint(pieces)
    print(len(pieces))
    g = make_graph(pieces)

    for vertex in sorted(g.vs, key=lambda vertex: vertex.attributes()["numbers"]):
        print(vertex.attributes()["numbers"], vertex.degree())
    igraph.plot(g, layout=g.layout("kk"), target="tridom.pdf")
    g.write_dot("tridom-neato.dot")
    render_igraph_neato(pathlib.Path("tridom-neato.dot"))


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

    left_edges = [(a, b), (b, c), (c, a)]
    right_edges = [(x, y), (y, z), (z, x)]
    for left_edge in left_edges:
        for right_edge in right_edges:
            print(left_edge, right_edge)
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
        for j, right in enumerate(pieces):
            if can_dock(left, right) and left != right:
                if not g.are_connected(str(left), str(right)):
                    g.add_edge(str(left), str(right))
    g.simplify()
    return g


if __name__ == "__main__":
    main()
