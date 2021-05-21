import numpy as np
import tqdm
import matplotlib.pyplot as pl

from . import game


def main():
    rewards = []
    for i in tqdm.tqdm(range(1000)):
        g = game.Game()
        reward = 0
        while not g.is_game_over():
            for dir in game.directions:
                old_board = np.array(g.board)
                reward += g.move(dir)
                if not np.all(old_board == g.board):
                    g.spawn()
                    break
        rewards.append(reward)

    print(rewards)
    pl.clf()
    pl.hist(rewards, bins="sturges")
    pl.title("Baseline with Simple Strategy")
    pl.xlabel("Score")
    pl.ylabel("Count")
    pl.tight_layout()
    pl.savefig("simple-strategy-rewards.png", dpi=300)
    pl.savefig("simple-strategy-rewards.svg")
    pl.savefig("simple-strategy-rewards.pdf")
