from typing import List
from typing import Optional

from .tiles import available_tiles
from .tiles import Direction
from .tiles import FitType
from .tiles import Tile
from .tiles import TransportType

board_size = 9
board: List[List[Optional[Tile]]] = []
for i in range(board_size):
    board.append([None] * board_size)


board[2][0] = Tile(
    "exit rail",
    "     \n     \n  #++\n     \n     ",
    [[(Direction.RIGHT, TransportType.RAIL)]],
)
board[4][0] = Tile(
    "exit road",
    "     \n     \n  #..\n     \n     ",
    [[(Direction.RIGHT, TransportType.ROAD)]],
)
board[6][0] = board[2][0]

board[0][2] = Tile(
    "exit road",
    "     \n     \n  #  \n  .  \n  .  ",
    [[(Direction.DOWN, TransportType.ROAD)]],
)
board[0][4] = Tile(
    "exit rail",
    "     \n     \n  #  \n  +  \n  +  ",
    [[(Direction.DOWN, TransportType.RAIL)]],
)
board[0][6] = board[0][2]

board[2][-1] = Tile(
    "exit rail",
    "     \n     \n++#  \n     \n     ",
    [[(Direction.LEFT, TransportType.RAIL)]],
)
board[4][-1] = Tile(
    "exit road",
    "     \n     \n..#  \n     \n     ",
    [[(Direction.LEFT, TransportType.ROAD)]],
)
board[6][-1] = board[2][-1]

board[-1][2] = Tile(
    "exit road",
    "  .  \n  .  \n  #  \n     \n     ",
    [[(Direction.UP, TransportType.ROAD)]],
)
board[-1][4] = Tile(
    "exit rail",
    "  +  \n  +  \n  #  \n     \n     ",
    [[(Direction.UP, TransportType.RAIL)]],
)
board[-1][6] = board[-1][2]

steps = 100


def print_board():
    result = []
    for i in range(board_size * 5):
        result.append([])

    for i in range(board_size):
        for j in range(board_size):
            for k in range(5):
                tile = board[i][j]
                row = tile.image[k] if tile else "     "
                result[i * 5 + k].append(row)

    print("@")
    for row in result:
        print("".join(row))

    global steps
    if steps < 0:
        raise RuntimeError()
    steps -= 1


def do_step():
    print_board()
    for tile in available_tiles:
        for i in range(1, board_size - 1):
            for j in range(1, board_size - 1):
                if board[i][j] is not None:
                    continue

                fits = [
                    tile.fits(board[i][j - 1], Direction.LEFT),
                    tile.fits(board[i][j + 1], Direction.RIGHT),
                    tile.fits(board[i - 1][j], Direction.UP),
                    tile.fits(board[i + 1][j], Direction.DOWN),
                ]

                if all(fit != FitType.INCOMPATIBLE for fit in fits) and any(
                    fit == FitType.MATCHES for fit in fits
                ):
                    board[i][j] = tile
                    do_step()
                    board[i][j] = None


def main():
    do_step()


if __name__ == "__main__":
    main()
