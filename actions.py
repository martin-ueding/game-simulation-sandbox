import copy
import typing


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