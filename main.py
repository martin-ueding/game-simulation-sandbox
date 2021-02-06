import copy
import pprint
import random

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
    'actions': [],
    'recruits': {},
}


def score(state: dict) -> int:
    score = 0

    num_achievements = len(state['achievements'])
    num_hexes = len(set(state['workers']) | set(state['buildings'].values()))
    num_resource_pairs = (state['grain'] + state['metal'] + state['oil'] + state['wood']) // 2

    if state['popularity'] <= 6:
        score += 3 * num_achievements
        score += 2 * num_hexes
        score += 1 * num_resource_pairs
    elif state['popularity'] <= 12:
        score += 4 * num_achievements
        score += 3 * num_hexes
        score += 2 * num_resource_pairs
    else:
        score += 5 * num_achievements
        score += 4 * num_hexes
        score += 3 * num_resource_pairs

    score += state['money']
    state['score'] = score
    return score


def beam_search_score(state: dict) -> int:
    score = 0

    score += 5 * len(state['buildings'])**2
    score += 5 * len(state['mechs'])**2
    score += 5 * len(state['recruits'])**2
    score += 4 * len(state['upgrades top'])**2

    score += 100 * len(state['achievements'])

    return score


def main():
    state = copy.deepcopy(initial_state)
    state['player mat'] = 'innovative'
    player_mat = player_mats.mats[state['player mat']]
    tree = state_tree.Tree(state)

    for round in range(1, 51):
        print('Round', round)
        pprint.pprint(tree.leaves[0])

        old_leaves = tree.leaves
        tree.reset_leaves()
        for leaf in old_leaves:
            state = leaf.state
            for column, (top_action, bottom_action) in enumerate(player_mat.actions):
                if column == state['last column']:
                    continue

                states_top = top_action(state)
                states_top.append(state)
                states_bottom = []
                for state_top in states_top:
                    states_bottom += bottom_action(state_top)
                states_bottom += states_top

                for new_state in states_bottom:
                    if new_state == state:
                        continue
                    new_state['last column'] = column
                    tree.insert(leaf, new_state)

        scores = sorted(set([score(node.state) for node in tree.leaves]))
        print('Number of leaves:', len(tree.leaves))
        print('Unique scores:', scores)
        print()
        tree.leaves.sort(key=lambda node: beam_search_score(node.state), reverse=True)
        tree.leaves = tree.leaves[:500] + random.sample(tree.leaves[500:], min(max(0, len(tree.leaves) - 500), 500))
        tree.leaves.sort(key=lambda node: beam_search_score(node.state), reverse=True)


    print()
    print('Top 10 beams:')
    for node in tree.leaves[:5]:
        print()
        pprint.pprint(node.state)


if __name__ == '__main__':
    main()
