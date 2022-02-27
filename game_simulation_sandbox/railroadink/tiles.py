import dataclasses
import enum
import pathlib
import random
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import PIL.Image


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

rotate_direction = {
    Direction.UP: Direction.RIGHT,
    Direction.RIGHT: Direction.DOWN,
    Direction.DOWN: Direction.LEFT,
    Direction.LEFT: Direction.UP,
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
        connections: List[List[Tuple[Direction, TransportType]]],
        image: Optional[np.ndarray] = None,
    ):
        self.name = name
        if image is not None:
            self.image = image
        else:
            self.image = np.array(
                PIL.Image.open(pathlib.Path(__file__).parent / "tiles" / f"{name}.png")
            )
        self.connections = [set(x) for x in connections]
        self.connections.sort()

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


def rotate_tile(tile: Tile) -> Tile:
    new_pixels = np.rot90(tile.image, -1)

    new_connections = []
    for connection in tile.connections:
        new_connection = []
        for direction, transport_type in connection:
            new_connection.append((rotate_direction[direction], transport_type))
        new_connections.append(new_connection)

    return Tile(tile.name, new_connections, new_pixels)


unique_tiles_twofold = [
    Tile(
        "straight rail",
        [[(Direction.LEFT, TransportType.RAIL), (Direction.RIGHT, TransportType.RAIL)]],
    ),
    Tile(
        "straight road",
        [[(Direction.LEFT, TransportType.ROAD), (Direction.RIGHT, TransportType.ROAD)]],
    ),
    Tile(
        "overpass",
        [
            [
                (Direction.LEFT, TransportType.RAIL),
                (Direction.RIGHT, TransportType.RAIL),
            ],
            [(Direction.UP, TransportType.ROAD), (Direction.DOWN, TransportType.ROAD)],
        ],
    ),
    Tile(
        "double road",
        [
            [(Direction.LEFT, TransportType.ROAD), (Direction.UP, TransportType.ROAD)],
            [
                (Direction.RIGHT, TransportType.ROAD),
                (Direction.DOWN, TransportType.ROAD),
            ],
        ],
    ),
    Tile(
        "double rail",
        [
            [(Direction.LEFT, TransportType.RAIL), (Direction.UP, TransportType.RAIL)],
            [
                (Direction.RIGHT, TransportType.RAIL),
                (Direction.DOWN, TransportType.RAIL),
            ],
        ],
    ),
]

unique_tiles_fourfold = [
    Tile(
        "curve road",
        [[(Direction.LEFT, TransportType.ROAD), (Direction.DOWN, TransportType.ROAD)]],
    ),
    Tile(
        "curve rail",
        [[(Direction.LEFT, TransportType.RAIL), (Direction.DOWN, TransportType.RAIL)]],
    ),
    Tile(
        "T road",
        [
            [
                (Direction.LEFT, TransportType.ROAD),
                (Direction.RIGHT, TransportType.ROAD),
                (Direction.DOWN, TransportType.ROAD),
            ]
        ],
    ),
    Tile(
        "T rail",
        [
            [
                (Direction.LEFT, TransportType.RAIL),
                (Direction.RIGHT, TransportType.RAIL),
                (Direction.DOWN, TransportType.RAIL),
            ]
        ],
    ),
    Tile(
        "curve station 1",
        [[(Direction.LEFT, TransportType.RAIL), (Direction.DOWN, TransportType.ROAD)]],
    ),
    Tile(
        "curve station 2",
        [[(Direction.LEFT, TransportType.ROAD), (Direction.DOWN, TransportType.RAIL)]],
    ),
    Tile(
        "straight station",
        [[(Direction.LEFT, TransportType.ROAD), (Direction.RIGHT, TransportType.RAIL)]],
    ),
    Tile(
        "T road station",
        [
            [
                (Direction.LEFT, TransportType.ROAD),
                (Direction.RIGHT, TransportType.ROAD),
                (Direction.DOWN, TransportType.RAIL),
            ]
        ],
    ),
    Tile(
        "T rail station",
        [
            [
                (Direction.LEFT, TransportType.RAIL),
                (Direction.RIGHT, TransportType.RAIL),
                (Direction.DOWN, TransportType.ROAD),
            ]
        ],
    ),
    Tile(
        "cul-de-sac road",
        [[(Direction.LEFT, TransportType.ROAD)]],
    ),
    Tile(
        "cul-de-sac rail",
        [[(Direction.LEFT, TransportType.RAIL)]],
    ),
]

available_tiles = []
for tile in unique_tiles_twofold:
    for i in range(2):
        available_tiles.append(tile)
        tile = rotate_tile(tile)
for tile in unique_tiles_fourfold:
    for i in range(4):
        available_tiles.append(tile)
        tile = rotate_tile(tile)


for tile in available_tiles:
    print(tile.name)

random.shuffle(available_tiles)
