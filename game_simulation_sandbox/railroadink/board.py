from typing import List
from typing import Optional

from .tiles import available_tiles
from .tiles import Direction
from .tiles import FitType
from .tiles import rotate_tile
from .tiles import Tile
from .tiles import TransportType

board_size = 9
board: List[List[Optional[Tile]]] = []
for i in range(board_size):
    board.append([None] * board_size)


exit_rail_right = Tile(
    "exit rail",
    "     \n     \n  #++\n     \n     ",
    [[(Direction.RIGHT, TransportType.RAIL)]],
)
exit_rail_down = rotate_tile(exit_rail_right)
exit_rail_left = rotate_tile(exit_rail_down)
exit_rail_up = rotate_tile(exit_rail_left)

exit_road_right = Tile(
    "exit road",
    "     \n     \n  #..\n     \n     ",
    [[(Direction.RIGHT, TransportType.ROAD)]],
)
exit_road_down = rotate_tile(exit_road_right)
exit_road_left = rotate_tile(exit_road_down)
exit_road_up = rotate_tile(exit_road_left)


board[2][0] = exit_rail_right
board[4][0] = exit_road_right
board[6][0] = exit_rail_right

board[0][2] = exit_rail_down
board[0][6] = exit_road_down

board[2][-1] = exit_rail_left
board[4][-1] = exit_road_left
board[6][-1] = exit_rail_left

board[-1][2] = exit_road_up
board[-1][4] = exit_rail_up
board[-1][6] = exit_road_up

steps = 2


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
                    print_board()
                    do_step()
                    board[i][j] = None
                    print_board()


def main():
    do_step()


if __name__ == "__main__":
    main()
