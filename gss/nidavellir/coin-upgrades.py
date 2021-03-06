import argparse
import collections
import copy
import itertools
import json
import pprint
import typing

import anytree
import tqdm


def make_bank(coins: list) -> typing.Dict[int, int]:
    bank = {}
    for value in range(5, 12):
        bank[value] = 3
    for value in range(12, 15):
        bank[value] = 2
    for value in range(15, 26):
        bank[value] = 1
    for coin in coins:
        if coin >= 5:
            bank[coin] -= 1
    return bank


def make_begin() -> list:
    return [2, 3, 4, 5]


def possible_upgrades(coins: list) -> list:
    new_states = []
    bank = make_bank(coins)
    for selected in itertools.combinations(coins, 2):
        keep = min(selected)
        new = min(selected) + max(selected)
        while new <= 25 and bank[new] == 0:
            new += 1
        if new <= 25:
            new_coins = copy.copy(coins)
            new_coins.remove(max(selected))
            new_coins.append(new)
            new_states.append(new_coins)
    return new_states


def main():
    options = _parse_args()

    state = make_begin()
    start = anytree.Node(state)
    print(anytree.RenderTree(start))

    leaves = [state]
    for round in range(8):
        old_leaves = leaves
        leaves = []
        for leaf in tqdm.tqdm(old_leaves):
            leaves += possible_upgrades(leaf)

    leaves.sort(key=lambda leaf: sum(leaf))
    pprint.pprint(leaves[:10], compact=True, width=100)

    bins = collections.defaultdict(list)
    for leaf in leaves:
        s = sum(leaf)
        bins[s].append(leaf)

    for score, states in sorted(bins.items()):
        print(score, len(states))
        pprint.pprint(states[:3])
        print()



def _parse_args():
    parser = argparse.ArgumentParser(description='')
    options = parser.parse_args()

    return options


if __name__ == '__main__':
    main()
