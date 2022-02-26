import dataclasses
import enum
from typing import List
from typing import Tuple


class Direction(enum.Enum):
    RIGHT
    UP
    LEFT
    DOWN


class Tile:
    image: str
    connections: List[Tuple[Direction]]


straight_rail = """

"""


tiles = {
    "straight_rail": straight_rail,
}
