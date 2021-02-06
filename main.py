import collections
import copy
import pprint

import player_mats
import state_tree


initial_state = {
    'popularity': 3,
    'money': 5,
    'power': 0,
    'combat cards': 2,
    'workers': [2, 4],
    'hero': 5,
    'mechs': [],
    'wood': 0,
    'metal': 0,
    'grain': 0,
    'oil': 0,
    'buildings': {},
    'upgrades top': [],
    'upgrades bottom': {'build': 0, 'deploy': 0, 'enlist': 0, 'upgrade': 0},
    'achievements': [],
    'last column': -1,
    'actions': []
}

Hex = collections.namedtuple('Hex', ['type', 'neighbors'])

board = [
    Hex('wood', [1, 3]),
    Hex('metal', [0, 2, 3, 4]),
    Hex('oil', [1, 4]),
    Hex('workers', [0, 1, 4]),
    Hex('grain', [1, 2, 3]),
    Hex('home', [3, 5]),
]


def main():
    state = copy.deepcopy(initial_state)
    player_mat = player_mats.innovative
    tree = state_tree.Tree(state)

    for round in range(1):
        old_leaves = tree.leaves
        tree.reset_leaves()
        for leaf in old_leaves:
            state = leaf.state
            for column, (top_action, bottom_action) in enumerate(player_mat):
                if column == state['last column']:
                    continue

                states_top = top_action(state)
                states_top.append(state)
                states_bottom = bottom_action(states_top)
                states_bottom += states_top

                new_states = [new_state for new_state in states_bottom if new_state != state]

                for new_state in states_bottom:
                    if new_state == state:
                        continue
                    new_state['last column'] = column
                    tree.insert(leaf, new_state)

        for node in tree.leaves:
            pprint.pprint(node.state)


if __name__ == '__main__':
    main()
