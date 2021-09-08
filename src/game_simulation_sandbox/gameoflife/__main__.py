import json
import pathlib
import subprocess

import click
import numpy as np

from . import transition

transition_path = pathlib.Path("game-of-life.json")


@click.group()
def cli():
    pass


@cli.command()
@click.option("--height", default=2, show_default=True)
@click.option("--width", default=2, show_default=True)
def gather(height: int, width: int) -> None:
    if transition_path.is_file():
        with open(transition_path) as f:
            transitions = json.load(f)
    else:
        transitions = {}

    for start in iter_states(height, width):
        print(to_string(start))
        state = transition.remove_padding(start)
        for i in range(10):
            if to_string(state) in transitions:
                break
            new_state = transition.next_state(state)
            transitions[to_string(state)] = to_string(new_state)
            state = new_state

    with open(transition_path, "w") as f:
        json.dump(transitions, f, indent=2, sort_keys=True)


def iter_states(height, width):
    for state_id in range(2 ** (height * width)):
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
    with open("game-of-life.dot", "w") as f:
        f.write("digraph {\n")
        f.write("overlap = false\n")
        f.write("splines = true\n")
        f.write('node [fontname = "Noto Mono"]\n')
        for source, destination in transitions.items():
            f.write(f'"{source}" -> "{destination}"'.replace("\n", "\\n") + "\n")
        f.write("}\n")
    subprocess.call(
        ["neato", "-T", "pdf", "-o", "game-of-life.pdf", "game-of-life.dot"]
    )


if __name__ == "__main__":
    main()
