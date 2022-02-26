import igraph

from . import roadmap


def test_simple() -> None:
    g = igraph.Graph()
    g.add_vertices(6)
    g.add_edges([(0, 1), (1, 2), (2, 3), (2, 4), (3, 5)])
    g.vs["label"] = [f"{i}" for i in g.vs.indices]
    layout = g.layout("kk")
    igraph.plot(g, layout=layout, target="test.pdf")
    path = roadmap.get_longest_simple_path(g)
    indices = [vertex.index for vertex in path]
    indices.sort()
    assert indices == [0, 1, 2, 3, 5]
