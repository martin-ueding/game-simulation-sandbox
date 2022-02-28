import copy
import pathlib
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import PIL

from ..treesearch.interface import Observer
from ..treesearch.interface import TreeIterator
from .tiles import available_tiles
from .tiles import Direction
from .tiles import exit_rail_down
from .tiles import exit_rail_left
from .tiles import exit_rail_right
from .tiles import exit_rail_up
from .tiles import exit_road_down
from .tiles import exit_road_left
from .tiles import exit_road_right
from .tiles import exit_road_up
from .tiles import FitType
from .tiles import Tile
from game_simulation_sandbox.treesearch.backtracking import Backtracking
from game_simulation_sandbox.treesearch.random_walk import RandomWalk


class Board:
    def __init__(self):
        self.available_tiles = available_tiles
        self.board_size = 9
        self.board: List[List[Optional[Tile]]] = []
        for i in range(self.board_size):
            self.board.append([None] * self.board_size)

        self.board[2][0] = exit_rail_right
        self.board[4][0] = exit_road_right
        self.board[6][0] = exit_rail_right

        self.board[0][2] = exit_road_up
        self.board[0][4] = exit_rail_up
        self.board[0][6] = exit_road_up

        self.board[2][-1] = exit_rail_left
        self.board[4][-1] = exit_road_left
        self.board[6][-1] = exit_rail_left

        self.board[-1][2] = exit_road_down
        self.board[-1][4] = exit_rail_down
        self.board[-1][6] = exit_road_down

    def get_open_positions(self) -> Generator[Tuple[int, int], None, None]:
        for i in range(1, self.board_size - 1):
            for j in range(1, self.board_size - 1):
                if self.board[i][j] is not None:
                    continue
                if (
                    has_direction(self.board[i - 1][j], Direction.DOWN)
                    or has_direction(self.board[i][j + 1], Direction.LEFT)
                    or has_direction(self.board[i][j - 1], Direction.RIGHT)
                    or has_direction(self.board[i + 1][j], Direction.UP)
                ):
                    yield i, j

    def replace(self, i: int, j: int, tile: Tile) -> "Board":
        result = copy.copy(self)
        result.board = [copy.copy(row) for row in self.board]
        result.board[i][j] = tile
        return result


def has_direction(tile: Tile, dir: Direction):
    if tile is None:
        return False
    else:
        for d, t in tile.open:
            if d == dir:
                return True
        return False


class RailroadInkIterator(TreeIterator):
    def __init__(self, board: Board):
        self.board = board

    def get_children(self) -> Generator["RailroadInkIterator", None, None]:
        board = self.board.board
        for i, j in self.board.get_open_positions():
            for tile in self.board.available_tiles:
                fits = [
                    tile.fits(board[i][j - 1], Direction.LEFT),
                    tile.fits(board[i][j + 1], Direction.RIGHT),
                    tile.fits(board[i - 1][j], Direction.UP),
                    tile.fits(board[i + 1][j], Direction.DOWN),
                ]

                if all(fit != FitType.INCOMPATIBLE for fit in fits) and any(
                    fit == FitType.MATCHES for fit in fits
                ):
                    yield RailroadInkIterator(self.board.replace(i, j, tile))


class VideoObserver(Observer):
    def __init__(self):
        self.steps = 0

        self.img_empty = np.ones((100, 100), np.uint8) * 255
        self.img_empty_tile = np.array(
            PIL.Image.open(pathlib.Path(__file__).parent / "tiles" / "empty tile.png")
        )
        self.img_open = np.array(
            PIL.Image.open(pathlib.Path(__file__).parent / "tiles" / "open.png")
        )

    def observe(self, state: TreeIterator) -> None:
        assert isinstance(state, RailroadInkIterator)

        board_size = state.board.board_size
        open_positions = set(state.board.get_open_positions())
        img_board = np.zeros((board_size * 100, board_size * 100), np.uint8)
        for i in range(board_size):
            for j in range(board_size):
                if state.board.board[i][j] is None:
                    if i == 0 or j == 0 or i == board_size - 1 or j == board_size - 1:
                        img = self.img_empty
                    elif (i, j) in open_positions:
                        img = self.img_open
                    else:
                        img = self.img_empty_tile
                else:
                    img = state.board.board[i][j].image
                img_board[i * 100 : (i + 1) * 100, j * 100 : (j + 1) * 100] = img

        pil_image = PIL.Image.fromarray(img_board)
        pil_image.save(f"railroad/{self.steps:06d}.png", "PNG")

        self.steps += 1
        if self.steps >= 1000:
            raise RuntimeError()


def main():
    pathlib.Path("railroad").mkdir(exist_ok=True)
    board = Board()
    iterator = RailroadInkIterator(board)
    observer = VideoObserver()
    tree_search = RandomWalk(observer)
    while True:
        tree_search.run(iterator)


if __name__ == "__main__":
    main()
