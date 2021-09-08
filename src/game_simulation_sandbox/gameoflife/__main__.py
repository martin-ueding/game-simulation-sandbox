import json
import pathlib
import subprocess

import click
import igraph
import numpy as np
from tqdm import tqdm

from . import transition
from .. import igraph_util

transition_path = pathlib.Path("game-of-life.json")


@click.group()
def cli():
    pass


@cli.command()
@click.option("--height", default=2, show_default=True)
@click.option("--width", default=2, show_default=True)
@click.option("--steps", default=10, show_default=True)
def gather(height: int, width: int, steps: int) -> None:
    if transition_path.is_file():
        with open(transition_path) as f:
            transitions = json.load(f)
    else:
        transitions = {}

    for start in iter_states(height, width):
        state = transition.remove_padding(start)
        for i in range(steps):
            if to_string(state) in transitions:
                break
            new_state = transition.next_state(state)
            transitions[to_string(state)] = to_string(new_state)
            state = new_state

    with open(transition_path, "w") as f:
        json.dump(transitions, f, indent=2, sort_keys=True)


def iter_states(height, width):
    for state_id in tqdm(range(2 ** (height * width))):
        state = np.zeros(height * width, dtype=int)
        for pos in range(len(state)):
            if state_id % 2 != 0:
                state[pos] = 1
            state_id //= 2
        yield state.reshape((height, width))


def to_string(state: np.ndarray) -> str:
    return "\n".join(
        "".join("█" if cell == 1 else "░" for cell in row) for row in state
    )


@cli.command()
def make_graph() -> None:
    with open(transition_path) as f:
        transitions = json.load(f)
    generate_graph(transitions)


def generate_graph(transitions, filename=pathlib.Path("game-of-life.dot")) -> None:
    with open(filename, "w") as f:
        f.write("digraph {\n")
        f.write("overlap = false\n")
        f.write("splines = true\n")
        f.write('node [fontname = "Noto Mono"]\n')
        for source, destination in transitions.items():
            f.write(f'"{source}" -> "{destination}"'.replace("\n", "\\n") + "\n")
        f.write("}\n")
    subprocess.call(
        ["neato", "-T", "pdf", "-o", filename.with_suffix(".pdf"), filename]
    )
    subprocess.call(
        ["neato", "-T", "png", "-o", filename.with_suffix(".png"), filename]
    )


@cli.command()
def make_graphs() -> None:
    with open(transition_path) as f:
        transitions = json.load(f)

    g = igraph.Graph()
    states = set(transitions.keys()) | set(transitions.values())
    for state in states:
        g.add_vertex(state, label=state)
    for source, destination in transitions.items():
        g.add_edge(source, destination)

    subgraphs = list(g.decompose())
    for i, s in tqdm(enumerate(subgraphs)):
        subgraph_states = [v.attributes()["name"] for v in s.vs]
        if "" in subgraph_states:
            continue
        subtransitions = {
            source: transitions[source]
            for source in subgraph_states
            if source in transitions
        }
        filename = pathlib.Path(f"subgraph-{i}.dot")
        generate_graph(subtransitions, filename)


if __name__ == "__main__":
    main()
