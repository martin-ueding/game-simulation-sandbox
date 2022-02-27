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

board[0][2] = exit_rail_down
board[0][6] = exit_road_down

board[2][-1] = exit_rail_left
board[4][-1] = exit_road_left
board[6][-1] = exit_rail_left

board[-1][2] = exit_road_up
board[-1][4] = exit_rail_up
board[-1][6] = exit_road_up


img_board = np.zeros((board_size * 100, board_size * 100, 4), np.uint8)
img_empty = np.array(
    PIL.Image.open(pathlib.Path(__file__).parent / "tiles" / "empty.png")
)
img_empty_tile = np.array(
    PIL.Image.open(pathlib.Path(__file__).parent / "tiles" / "empty tile.png")
)

for i in range(board_size):
    for j in range(board_size):
        if board[i][j] is None:
            if i == 0 or j == 0 or i == board_size - 1 or j == board_size - 1:
                img = img_empty
            else:
                img = img_empty_tile
        else:
            img = board[i][j].image
        img_board[i * 100 : (i + 1) * 100, j * 100 : (j + 1) * 100] = img


steps = 0


def print_board():
    global steps
    pil_image = PIL.Image.fromarray(img_board)
    pil_image.save(f"test-{steps:06d}.png", "PNG")

    if steps >= 0:
        raise RuntimeError()
    steps += 1


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
