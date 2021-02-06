import collections

Hex = collections.namedtuple('Hex', ['type', 'neighbors'])

board = [
    Hex('wood', [1, 3]),
    Hex('metal', [0, 2, 3, 4]),
    Hex('oil', [1, 4]),
    Hex('workers', [0, 1, 4]),
    Hex('grain', [1, 2, 3]),
    Hex('home', [3, 5]),
]