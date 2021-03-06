#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2021 Martin Ueding <mu@martin-ueding.de>

import argparse
import copy
import itertools
import json
import pprint
import typing

import anytree
import tdqm


def make_bank() -> typing.Dict[int, int]:
    bank = {}
    for value in range(5, 12):
        bank[value] = 3
    for value in range(12, 15):
        bank[value] = 2
    for value in range(15, 26):
        bank[value] = 1
    return bank


def make_begin() -> list:
    return [2, 3, 4, 5]


def possible_upgrades(coins: list, bank: dict) -> list:
    new_states = []
    for selected in itertools.combinations(coins, 2):
        keep = min(selected)
        new = min(selected) + max(selected)
        while bank[new] == 0 and new <= 25:
            new += 1
        if new <= 25:
            new_bank = copy.copy(bank)
            new_bank[new] -= 1
            new_coins = copy.copy(coins)
            new_coins.remove(max(selected))
            new_coins.append(new)
            new_states.append(dict(coins=new_coins, bank=new_bank))
    return new_states


def main():
    options = _parse_args()

    state = dict(coins=make_begin(), bank=make_bank())
    start = anytree.Node(state)
    print(anytree.RenderTree(start))

    leaves = [start]
    for round in range(8):
        old_leaves = leaves
        leaves = []
        for leaf in tqdm.tqdm(old_leaves):
            leaves += possible_upgrades(leaf['coins'], leaf['bank'])


def _parse_args():
    parser = argparse.ArgumentParser(description='')
    options = parser.parse_args()

    return options


if __name__ == '__main__':
    main()
