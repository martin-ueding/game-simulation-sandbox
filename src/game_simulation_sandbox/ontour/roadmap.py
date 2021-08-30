import copy
import pathlib
import random
from typing import *

import igraph
import yaml


def build_undirected_graph(path: pathlib.Path) -> igraph.Graph:
    this_path = pathlib.Path(__file__)
    data_path = this_path.parent / "data"
    graph_path = data_path / "usa.yml"
    with open(graph_path) as f:
        graph_usa = yaml.safe_load(f)

    g = igraph.Graph()
    g.add_vertex("AZ")

    for source, destinations in graph_usa.items():
        if source not in g.vs["name"]:
            g.add_vertex(source)
        for destination in destinations.split(" "):
            if destination not in g.vs["name"]:
                g.add_vertex(destination)
            start, end = sorted([source, destination])

            g.add_edge(source, destination)

    g.simplify()
    return g


def set_random_numbers(graph: igraph.Graph) -> None:
    random.seed(0)
    numbers = list(range(len(graph.vs)))
    random.shuffle(numbers)
    graph.vs["number"] = numbers


def get_longest_simple_path(graph: igraph.Graph) -> List[int]:
    pass


def make_neato() -> None:
    g.vs["label"] = g.vs["name"]
    print(g)
    d = g.as_directed()
    print(d)
    layout = d.layout("kk")
    igraph.plot(d, layout=layout, target="usa-graph.pdf")

    d.vs["label"] = [
        f"{name} {number}" for name, number in zip(d.vs["name"], d.vs["number"])
    ]

    to_delete = []
    for edge in d.es:
        if d.vs[edge.source]["number"] > d.vs[edge.target]["number"]:
            to_delete.append(edge)

    d.delete_edges(to_delete)
    igraph.plot(d, layout=layout, target="usa-grap2.pdf")
    d.write_dot("usa-neato2.dot")
    igraph.plot(d, layout=layout, target="usa-grap2.pdf")


if __name__ == "__main__":
    make_neato()
