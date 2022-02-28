import random

from .interface import Observer
from .interface import TreeIterator
from .interface import TreeSearch


class RandomWalk(TreeSearch):
    def __init__(self, observer: Observer):
        self.observer = observer

    def run(self, start: TreeIterator) -> None:
        state = start
        while True:
            self.observer.observe(state)
            children = list(state.get_children())
            if len(children) == 0:
                return
            state = random.choice(children)
