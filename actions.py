import collections
import copy
import itertools
import typing

import board
import player_mats


def increase_popularity(state: dict, amount) -> None:
    state['popularity'] = min(18, state['popularity'] + amount)
    if state['popularity'] == 18 and '18 popularity' not in state['achievements']:
        state['achievements'].append('18 popularity')


def increase_power(state: dict, amount) -> None:
    state['power'] = min(16, state['power'] + amount)
    if state['power'] == 16 and '16 power' not in state['achievements']:
        state['achievements'].append('16 power')


def bolster(state: dict) -> typing.List[dict]:
    # We need to pay one money.
    if state['money'] == 0:
        return []
    state = copy.deepcopy(state)
    state['money'] -= 1

    if 'monument' in state['buildings']:
        increase_popularity(state, 1)

    results = []

    # We can choose to invest into our power.
    if state['power'] < 16:
        state_power = copy.deepcopy(state)
        amount = 2
        if 'power' in state_power['upgrades top']:
            amount += 1
        increase_power(state_power, amount)
        state_power['actions'][-1][0] = 'Bolster power'

    # Alternatively we can invest into combat cards.
    state_card = copy.deepcopy(state)
    state_card['combat cards'] += 1
    if 'combat cards' in state_card['upgrades top']:
        state_card['combat cards'] += 1
    state_card['actions'][-1][0] = 'Bolster combat cards'
    # In this simulation, combat cards have no point. We only want to perform this action when we get some popularity out of it.
    if 'monument' in state['buildings'] and state['popularity'] < 18:
        results.append(state_card)

    return results


def trade(state: dict) -> typing.List[dict]:
    # We need to pay one money.
    if state['money'] == 0:
        return []
    state = copy.deepcopy(state)
    state['money'] -= 1

    if 'armory' in state['buildings']:
        increase_power(state, 1)

    final_states = []

    # We can choose to buy any two resources.
    types = ['grain', 'metal', 'oil', 'wood']
    for i, res_1 in enumerate(types):
        for res_2 in types[i:]:
            state_resource = copy.deepcopy(state)
            state_resource[res_1] += 1
            state_resource[res_2] += 1
            state_resource['actions'][-1][0] = 'Buy {} and {}'.format(res_1, res_2)
            final_states.append(state_resource)

    # Alternatively we can invest into our popularity
    state_popularity = copy.deepcopy(state)
    amount = 1
    if 'popularity' in state_popularity['upgrades top']:
        amount += 1
    increase_popularity(state_popularity,  amount)
    state_popularity['actions'][-1][0] = 'Buy popularity'

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

    # The mill will create one unit of resources.
    if 'mill' in state['buildings']:
        mill_type = board.board[state['buildings']['mill']].type
        # If the mill is build in a village, it produces one more worker in that hex.
        if mill_type == 'worker' and len(state['worker']) < 8:
            state['worker'].append(state['buildings']['mill'])
        # Otherwise it just produces one of the resources.
        else:
            state[mill_type] += 1

    num_production_hexes = 3 if 'production' in state['upgrades top'] else 2
    workers = collections.defaultdict(lambda: 0)
    for pos in state['workers']:
        if board.board[pos].type in ['grain', 'metal', 'oil', 'wood']:
            workers[pos] += 1

    final_states = []
    for hex_ids in itertools.combinations(workers.keys(), num_production_hexes):
        final_state = copy.deepcopy(state)
        strings = []
        for hex_id in hex_ids:
            hex_type = board.board[hex_id].type
            add = workers[hex_id]
            final_state[hex_type] += add
            strings.append('{} {} on H{}'.format(add, hex_type, hex_id))
        final_state['actions'][-1][0] = 'Produce {}'.format(' and '.join(strings))
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
                labels.append('Move ' + ', '.join('H{} ({}) → H{} ({})'.format(worker, board.board[worker].type, pos, board.board[pos].type) for worker, pos in zip(workers, new_pos)))

    for new_position, label in zip(new_positions, labels):
        new_state = copy.deepcopy(state)
        new_state['workers'] = new_position
        new_state['actions'][-1][0] = label
        final_states.append(new_state)

    state_money = copy.deepcopy(state)
    state_money['money'] += 1
    if 'money' in state_money['upgrades top']:
        state_money['money'] += 1
    state_money['actions'][-1][0] = 'Get money'
    final_states.append(state_money)

    return final_states


def build(state: dict) -> typing.List[dict]:
    fixed_costs, variable_cost, reward = player_mats.mats[state['player mat']].build
    cost = fixed_costs + variable_cost - state['upgrades bottom']['build']
    if cost > state['wood']:
        return []

    buildings_left = {'armory', 'monument', 'tunnel', 'mill'} - set(state['buildings'].keys())
    if len(buildings_left) == 0:
        return []

    final_states = []
    for building in buildings_left:
        for hex in set(state['workers']):
            # If there already is a building on that hex, we can't build.
            if hex in state['buildings'].values():
                continue
            if building == 'mill' and board.board[hex].type not in ['grain', 'metal', 'oil', 'wood', 'worker']:
                continue
            final_state = copy.deepcopy(state)
            final_state['wood'] -= cost
            final_state['buildings'][building] = hex
            final_state['money'] += reward
            if 'build' in final_state['recruits'].keys() and final_state['popularity'] <= 18:
                final_state['popularity'] += 1
            if len(final_state['buildings']) == 4:
                final_state['achievements'].append('all buildings')
            final_state['actions'][-1][1] = 'Build {} on H{}'.format(building, hex)
            final_states.append(final_state)

    return final_states


def deploy(state: dict) -> typing.List[dict]:
    fixed_costs, variable_cost, reward = player_mats.mats[state['player mat']].build
    cost = fixed_costs + variable_cost - state['upgrades bottom']['deploy']
    if cost > state['metal']:
        return []

    # If we already have all the mechs, there is nothing we can deploy.
    if len(state['mechs']) == 4:
        return []

    final_states = []
    for hex in set(state['workers']):
        final_state = copy.deepcopy(state)
        final_state['metal'] -= cost
        final_state['mechs'].append(hex)
        final_state['money'] += reward
        if 'deploy' in final_state['recruits'].keys():
            final_state['money'] += 1
        if len(final_state['mechs']) == 4:
            final_state['achievements'].append('all mechs')
        final_state['actions'][-1][1] = 'Deploy mech at H{}'.format(hex)
        final_states.append(final_state)

    return final_states


def enlist(state: dict) -> typing.List[dict]:
    fixed_costs, variable_cost, reward = player_mats.mats[state['player mat']].enlist
    cost = fixed_costs + variable_cost - state['upgrades bottom']['enlist']
    if cost > state['grain']:
        return []

    recruits_left = {'upgrade', 'deploy', 'build', 'enlist'} - set(state['recruits'].keys())
    bonuses_left = {'money', 'power', 'popularity', 'combat cards'} - set(state['recruits'].values())
    if len(recruits_left) == 0:
        return []

    final_states = []
    for recruit in recruits_left:
        for bonus in bonuses_left:
            final_state = copy.deepcopy(state)
            final_state['grain'] -= cost
            final_state['recruits'][recruit] = bonus
            if bonus == 'popularity':
                increase_popularity(final_state, 2)
            elif bonus == 'power':
                increase_power(final_state, 2)
            else:
                final_state[bonus] += 2
            final_state['money'] += reward
            if 'enlist' in state['recruits'].keys():
                final_state['combat cards'] += 1
            if len(final_state['recruits']) == 4:
                final_state['achievements'].append('all recruits')
            final_state['actions'][-1][1] = 'Enlist {} for {}'.format(recruit, bonus)
            final_states.append(final_state)

    return final_states


def upgrade(state: dict) -> typing.List[dict]:
    fixed_costs, variable_cost, reward = player_mats.mats[state['player mat']].upgrade
    cost = fixed_costs + variable_cost - state['upgrades bottom']['upgrade']
    if cost > state['oil']:
        return []

    upgrades_top_left = {'power', 'combat cards', 'popularity', 'production', 'movement', 'money'} - set(state['upgrades top'])
    if len(upgrades_top_left) == 0:
        return []
    upgrades_bottom_left = []
    for upgrade_bottom_type in {'upgrade', 'deploy', 'build', 'enlist'}:
        if state['upgrades bottom'][upgrade_bottom_type] < getattr(player_mats.mats[state['player mat']], upgrade_bottom_type)[2]:
            upgrades_bottom_left.append(upgrade_bottom_type)

    final_states = []
    for upgrade_top in upgrades_top_left:
        for upgrade_bottom in upgrades_bottom_left:
            final_state = copy.deepcopy(state)
            final_state['oil'] -= cost
            final_state['upgrades top'].append(upgrade_top)
            final_state['upgrades bottom'][upgrade_bottom] += 1
            final_state['money'] += reward
            if 'upgrade' in state['recruits'].keys() and final_state['power'] < 16:
                final_state['power'] += 1
            if len(final_state['upgrades top']) == 6:
                final_state['achievements'].append('all upgrades')
            final_state['actions'][-1][1] = 'Upgrade {} and {}'.format(upgrade_top, upgrade_bottom)
            final_states.append(final_state)

    return final_states