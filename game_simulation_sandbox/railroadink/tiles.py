import dataclasses
import enum
from typing import List
from typing import Optional
from typing import Tuple


class Direction(enum.Enum):
    RIGHT = 1
    UP = 2
    LEFT = 3
    DOWN = 4


opposite_directions = {
    Direction.RIGHT: Direction.LEFT,
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
}


class TransportType(enum.Enum):
    RAIL = 1
    ROAD = 2


class FitType(enum.Enum):
    MATCHES = 1
    OPEN = 2
    INCOMPATIBLE = 3


class Tile:
    def __init__(
        self,
        name: str,
        image: str,
        connections: List[List[Tuple[Direction, TransportType]]],
    ):
        self.name = name
        self.image = image.split("\n")
        self.connections = connections

        assert len(self.image) == 5
        assert all(len(row) == 5 for row in self.image)

    def fits(
        self,
        other: Optional["Tile"],
        direction_from_self: Direction,
    ) -> FitType:
        if other is None:
            return FitType.OPEN

        our_outgoing = None
        for connection in self.connections:
            for vertex in connection:
                if vertex[0] == direction_from_self:
                    our_outgoing = vertex[1]

        their_outgoing = None
        for connection in other.connections:
            for vertex in connection:
                if vertex[0] == opposite_directions[direction_from_self]:
                    their_outgoing = vertex[1]

        if our_outgoing is None and their_outgoing is None:
            result = FitType.OPEN
        elif our_outgoing == their_outgoing:
            result = FitType.MATCHES
        else:
            result = FitType.INCOMPATIBLE
        # print(f"Fit: {self}, {other}, {direction_from_self}: {result}")
        return result

    def __str__(self) -> str:
        return self.name


available_tiles = [
    Tile(
        "straight rail",
        "     \n     \n+++++\n     \n     ",
        [[(Direction.LEFT, TransportType.RAIL), (Direction.RIGHT, TransportType.RAIL)]],
    ),
    Tile(
        "straight road",
        "     \n     \n.....\n     \n     ",
        [[(Direction.LEFT, TransportType.ROAD), (Direction.RIGHT, TransportType.ROAD)]],
    ),
]
