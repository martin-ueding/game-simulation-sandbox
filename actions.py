import collections
import copy
import itertools
import typing

import board
import player_mats


def empty(state: dict) -> typing.List[dict]:
    return []


def bolster(state: dict) -> typing.List[dict]:
    # We need to pay one money.
    if state['money'] == 0:
        return []
    state = copy.deepcopy(state)
    state['money'] -= 1

    if 'monument' in state['buildings']:
        state['popularity'] += 1

    # We can choose to invest into our power.
    state_power = copy.deepcopy(state)
    state_power['power'] += 2
    if 'power' in state_power['upgrades top']:
        state_power['power'] += 1
    state_power['actions'].append('Bolster power')

    # Alternatively we can invest into combat cards.
    state_card = copy.deepcopy(state)
    state_card['combat cards'] += 1
    if 'combat cards' in state_card['upgrades top']:
        state_card['combat cards'] += 1
    state_card['actions'].append('Bolster combat cards')

    return [state_power, state_card]


def trade(state: dict) -> typing.List[dict]:
    # We need to pay one money.
    if state['money'] == 0:
        return []
    state = copy.deepcopy(state)
    state['money'] -= 1

    if 'armory' in state['buildings']:
        state['power'] += 1

    final_states = []

    # We can choose to buy any two resources.
    types = ['grain', 'metal', 'oil', 'wood']
    for i, res_1 in enumerate(types):
        for res_2 in types[i:]:
            state_resource = copy.deepcopy(state)
            state_resource[res_1] += 1
            state_resource[res_2] += 1
            state_resource['actions'].append('Buy {} and {}'.format(res_1, res_2))
            final_states.append(state_resource)

    # Alternatively we can invest into our popularity
    state_popularity = copy.deepcopy(state)
    state_popularity['popularity'] += 1
    if 'popularity' in state_popularity['upgrades top']:
        state_popularity['popularity'] += 1
    state_popularity['actions'].append('Buy popularity')

    return final_states + [state_popularity]


def produce(state: dict) -> typing.List[dict]:
    # Production has costs depending on the number of workers.
    cost = {}
    if len(state['workers']) >= 4:
        cost['power'] = 1
    if len(state['workers']) >= 6:
        cost['popularity'] = 1
    if len(state['workers']) >= 8:
        cost['money'] = 1

    # The player must be able to pay for the cost in order to perform this action.
    for cost_type, amount in cost.items():
        if state[cost_type] < amount:
            return []

    # The costs are deducted from the state.
    state = copy.deepcopy(state)
    for cost_type, amount in cost.items():
        state[cost_type] -= amount

    num_production_hexes = 3 if 'production' in state['upgrades top'] else 2
    workers = collections.defaultdict(lambda: 0)
    for pos in state['workers']:
        if board.board[pos].type in ['grain', 'metal', 'oil', 'wood']:
            workers[pos] += 1

    final_states = []
    for hex_ids in itertools.combinations(workers.keys(), num_production_hexes):
        final_state = copy.deepcopy(state)
        for hex_id in hex_ids:
            hex_type = board.board[hex_id].type
            add = workers[hex_id]
            final_state[hex_type] += add
        final_state['actions'].append('Produce on H{}'.format(str(hex_ids)))
        final_states.append(final_state)

    return final_states


def move(state: dict) -> typing.List[dict]:
    final_states = []

    new_positions = []
    labels = []
    num_movements = 3 if 'movement' in state['upgrades top'] else 2
    for workers in itertools.combinations(range(len(state['workers'])), num_movements):
        neighbors = [board.board[worker].neighbors for worker in workers]
        for new_pos in itertools.product(*neighbors):
            positions = copy.deepcopy(state['workers'])
            for worker, pos in zip(workers, new_pos):
                positions[worker] = pos
            positions.sort()
            if not positions  in new_positions:
                new_positions.append(positions)
                labels.append('Move ' + ', '.join('H{} → H{}'.format(worker, pos) for worker, pos in zip(workers, new_pos)))

    for new_position, label in zip(new_positions, labels):
        new_state = copy.deepcopy(state)
        new_state['workers'] = new_position
        new_state['actions'].append(label)
        final_states.append(new_state)

    state_money = copy.deepcopy(state)
    state_money['money'] += 1
    if 'money' in state_money['upgrades top']:
        state_money['money'] += 1
    state_money['actions'].append('Get money')
    final_states.append(state_money)

    return final_states


def build(state: dict) -> typing.List[dict]:
    fixed_costs, variable_cost, reward = player_mats.mats[state['player mat']].build
    cost = fixed_costs + variable_cost - state['upgrades bottom']['build']
    if cost > state['wood']:
        return []

    buildings_left = set(['armory', 'monument', 'tunnel', 'mill']) - set(state['buildings'].keys())
    if len(buildings_left) == 0:
        return []

    final_states = []
    for building in buildings_left:
        for hex in set(state['workers']):
            # If there already is a building on that hex, we can't build.
            if hex in state['buildings'].values():
                continue
            final_state = copy.deepcopy(state)
            final_state['buildings'][building] = hex
            final_states.append(final_state)

    return final_states