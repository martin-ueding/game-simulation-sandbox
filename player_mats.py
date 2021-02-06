import collections

import actions

PlayerMat = collections.namedtuple('PlayerMat', ['actions', 'upgrade', 'deploy', 'build', 'enlist'])
Costs = collections.namedtuple('Costs', ['fixed', 'variable', 'reward'])

mats = {'innovative': PlayerMat(
        actions=[
            (actions.trade, actions.empty),
            (actions.produce, actions.empty),
            (actions.bolster, actions.build),
            (actions.move, actions.empty),
        ],
        upgrade=(3, 0, 3),
        deploy=(2, 1, 1),
        build=(3, 1, 2),
        enlist=(1, 2, 0),
    )
}
