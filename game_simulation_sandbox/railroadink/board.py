import pathlib
from typing import List
from typing import Optional

import numpy as np
import PIL

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
    [[(Direction.RIGHT, TransportType.RAIL)]],
)
exit_rail_up = rotate_tile(exit_rail_right)
exit_rail_left = rotate_tile(exit_rail_up)
exit_rail_down = rotate_tile(exit_rail_left)

exit_road_right = Tile(
    "exit road",
    [[(Direction.RIGHT, TransportType.ROAD)]],
)
exit_road_up = rotate_tile(exit_road_right)
exit_road_left = rotate_tile(exit_road_up)
exit_road_down = rotate_tile(exit_road_left)


board[2][0] = exit_rail_right
board[4][0] = exit_road_right
board[6][0] = exit_rail_right

board[0][2] = exit_road_up
board[0][4] = exit_rail_up
board[0][6] = exit_road_up

board[2][-1] = exit_rail_left
board[4][-1] = exit_road_left
board[6][-1] = exit_rail_left

board[-1][2] = exit_road_down
board[-1][4] = exit_rail_down
board[-1][6] = exit_road_down


img_board = np.zeros((board_size * 100, board_size * 100), np.uint8)
img_empty = np.ones((100, 100), np.uint8) * 255
img_empty_tile = np.array(
    PIL.Image.open(pathlib.Path(__file__).parent / "tiles" / "empty tile.png")
)
img_open = np.array(
    PIL.Image.open(pathlib.Path(__file__).parent / "tiles" / "open.png")
)


steps = 0


def print_board():
    global steps
    pil_image = PIL.Image.fromarray(img_board)
    pil_image.save(f"railroad/{steps:06d}.png", "PNG")

    if steps >= 10000:
        raise RuntimeError()
    steps += 1


def has_direction(tile, dir):
    if tile is None:
        return False
    else:
        for d, t in tile.open:
            if d == dir:
                return True
        return False


def get_open(board):
    result = set()
    for i in range(1, board_size - 1):
        for j in range(1, board_size - 1):
            if board[i][j] is not None:
                continue
            if (
                has_direction(board[i - 1][j], Direction.DOWN)
                or has_direction(board[i][j + 1], Direction.LEFT)
                or has_direction(board[i][j - 1], Direction.RIGHT)
                or has_direction(board[i + 1][j], Direction.UP)
            ):
                result.add((i, j))

            img_board[i * 100 : (i + 1) * 100, j * 100 : (j + 1) * 100] = (
                img_open if (i, j) in result else img_empty_tile
            )
    return result


open_positions = get_open(board)
print(open_positions)

for i in range(board_size):
    for j in range(board_size):
        if board[i][j] is None:
            if i == 0 or j == 0 or i == board_size - 1 or j == board_size - 1:
                img = img_empty
            elif (i, j) in open_positions:
                img = img_open
            else:
                img = img_empty_tile
        else:
            img = board[i][j].image
        img_board[i * 100 : (i + 1) * 100, j * 100 : (j + 1) * 100] = img


def do_step(i, j):
    if i <= 0 or j <= 0 or i >= board_size - 1 or j >= board_size - 1:
        return
    if board[i][j] is not None:
        return

    for tile in available_tiles:
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
            img_board[i * 100 : (i + 1) * 100, j * 100 : (j + 1) * 100] = tile.image
            open_spots = get_open(board)
            print_board()
            for ii, jj in sorted(open_spots):
                do_step(ii, jj)
            board[i][j] = None
            img_board[i * 100 : (i + 1) * 100, j * 100 : (j + 1) * 100] = img_empty_tile
            get_open(board)
            print_board()


def main():
    do_step(4, 1)


if __name__ == "__main__":
    main()
