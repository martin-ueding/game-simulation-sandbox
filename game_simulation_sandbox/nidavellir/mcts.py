import random

import anytree
import click
import pandas as pd
from anytree.exporter import DotExporter
from tqdm import tqdm

from .coin_upgrades import make_begin
from .coin_upgrades import make_name
from .coin_upgrades import possible_upgrades


@click.command()
def main():
    state = make_begin()
    start = anytree.Node(make_name(state), coins=state, max_score=sum(state))
    print(anytree.RenderTree(start))

    performance = []

    for i in tqdm(range(100000)):
        cur = start
        for j in range(8):
            if not cur.children:
                cur.children = [
                    anytree.Node(make_name(state), coins=state, max_score=sum(state))
                    for state in possible_upgrades(cur.coins)
                ]
            cur = random.choices(
                cur.children, weights=[node.max_score for node in cur.children]
            )[0]

        while cur.parent:
            cur.parent.max_score = max(cur.max_score, cur.parent.max_score)
            cur = cur.parent

        performance.append((i, start.max_score))
    print(start)

    df = pd.DataFrame(performance, columns=["Step", "Max Score"])
    df.to_json("performance.js")

    DotExporter(start).to_dotfile("tree.dot")


if __name__ == "__main__":
    main()
