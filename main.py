import collections
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
    score = state['score']

    score += 5 * sum(action[1] is not None for action in state['actions'][-6:])

    score += 4 * len(state['buildings'])**2
    score += 4 * len(state['mechs'])**2
    score += 4 * len(state['recruits'])**2
    score += 4 * len(state['upgrades top'])**2
    score += 4 * len(state['workers'])**2

    score += 5 * len(state['achievements'])
    state['score search'] = score
    return score


def state_filter(state: dict) -> bool:
    if len(state['actions']) < 8:
        return True

    # I don't want to keep beams where the bottom row action hasn't been performed at least a few times in the last rounds.
    if sum(action[1] is not None for action in state['actions'][-8:]) < 2:
        return False

    # There is no point in having resources when there is nothing to use them for.
    if len(state['buildings']) == 4 and state['wood'] > 1:
        return False
    if len(state['mechs']) == 4 and state['metal'] > 1:
        return False
    if len(state['recruits']) == 4 and state['grain'] > 1:
        return False
    if len(state['upgrades top']) == 6 and state['oil'] > 1:
        return False

    return True


def main():
    state = copy.deepcopy(initial_state)
    state['player mat'] = 'innovative'
    player_mat = player_mats.mats[state['player mat']]
    tree = state_tree.Tree(state)


    finished = []

    for round in range(1, 26):
        print('Round', round)

        scores = sorted(set([score(node.state) for node in tree.leaves]))
        tree.leaves = [node for node in tree.leaves if state_filter(node.state)]
        print('Number of leaves filtered filter:', len(tree.leaves))

        top_k = 1500
        sampled = 500
        tree.leaves.sort(key=lambda node: beam_search_score(node.state), reverse=True)
        tree.leaves = tree.leaves[:top_k] + random.sample(tree.leaves[top_k:], min(max(0, len(tree.leaves) - top_k), sampled))
        tree.leaves.sort(key=lambda node: beam_search_score(node.state), reverse=True)

        if len(tree.leaves) == 0:
            break

        print('Top beam:')
        pprint.pprint(tree.leaves[0].state)

        old_leaves = tree.leaves
        tree.reset_leaves()
        for leaf in old_leaves:
            state = copy.deepcopy(leaf.state)
            state['actions'].append([None, None])
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
        print('Increase in leaves: {:.1f}'.format(len(tree.leaves) / len(old_leaves)))
        print('Unique scores:', scores)
        print('Finished beams:', len(finished))
        print()

        finished_ids = [i for i, node in enumerate(tree.leaves) if len(node.state['achievements']) == 6]
        for i in reversed(finished_ids):
            finished.append(tree.leaves[i])
            del tree.leaves[i]

    finished.sort(key=lambda node: score(node.state), reverse=True)
    tree.leaves.sort(key=lambda node: score(node.state), reverse=True)

    print()
    print('Top beams:')
    for node in tree.leaves[:5]:
        print()
        pprint.pprint(node.state)

    print()
    print('Top finished:')
    for node in finished[:5]:
        print()
        pprint.pprint(node.state)


if __name__ == '__main__':
    main()
