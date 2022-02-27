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


def rotate_tile(tile: Tile) -> Tile:
    pixels = [list(row) for row in tile.image]
    new_pixels = [[""] * 5 for i in range(5)]
    for i in range(5):
        for j in range(5):
            new_pixels[j][4 - i] = pixels[i][j]
    new_image = ["".join(row) for row in new_pixels]

    new_connections = []
    for connection in tile.connections:
        new_connection = []
        for direction, transport_type in connection:
            new_connection.append((rotate_direction[direction], transport_type))
        new_connections.append(new_connection)

    return Tile(tile.name, "\n".join(new_image), new_connections)


unique_tiles = [
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
    Tile(
        "curve road",
        "     \n     \n...  \n  .  \n  .  ",
        [[(Direction.LEFT, TransportType.ROAD), (Direction.DOWN, TransportType.ROAD)]],
    ),
    Tile(
        "curve rail",
        "     \n     \n+++  \n  +  \n  +  ",
        [[(Direction.LEFT, TransportType.RAIL), (Direction.DOWN, TransportType.RAIL)]],
    ),
    Tile(
        "T road",
        "     \n     \n.....\n  .  \n  .  ",
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
        "     \n     \n+++++\n  +  \n  +  ",
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
        "     \n     \n++#  \n  .  \n  .  ",
        [[(Direction.LEFT, TransportType.RAIL), (Direction.DOWN, TransportType.ROAD)]],
    ),
    Tile(
        "curve station 2",
        "     \n     \n..#  \n  +  \n  +  ",
        [[(Direction.LEFT, TransportType.ROAD), (Direction.DOWN, TransportType.RAIL)]],
    ),
    Tile(
        "straight station",
        "     \n     \n..#++\n     \n     ",
        [[(Direction.LEFT, TransportType.ROAD), (Direction.RIGHT, TransportType.RAIL)]],
    ),
    Tile(
        "T road station",
        "     \n     \n..#..\n  +  \n  +  ",
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
        "     \n     \n++#++\n  .  \n  .  ",
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
        "     \n     \n..#  \n     \n     ",
        [[(Direction.LEFT, TransportType.ROAD)]],
    ),
    Tile(
        "cul-de-sac rail",
        "     \n     \n++#  \n     \n     ",
        [[(Direction.LEFT, TransportType.RAIL)]],
    ),
    Tile(
        "overpass",
        "  .  \n  .  \n++.++\n  .  \n  .  ",
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
        "  .  \n .   \n.   .\n   . \n  .  ",
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
        "  +  \n +   \n+   +\n   + \n  +  ",
        [
            [(Direction.LEFT, TransportType.RAIL), (Direction.UP, TransportType.RAIL)],
            [
                (Direction.RIGHT, TransportType.RAIL),
                (Direction.DOWN, TransportType.RAIL),
            ],
        ],
    ),
]

available_tiles = []
for tile in unique_tiles:
    for i in range(4):
        if tile.image not in [t.image for t in available_tiles]:
            available_tiles.append(tile)
        tile = rotate_tile(tile)


for tile in available_tiles:
    print(tile.name, tile.connections)
    for line in tile.image:
        print(line)
