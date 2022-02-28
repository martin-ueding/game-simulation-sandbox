from .interface import Observer
from .interface import TreeIterator
from .interface import TreeSearch


class StackEntry:
    def __init__(self, state: TreeIterator):
        self.state = state
        self.child_iterator = iter(state.get_children())


class Backtracking(TreeSearch):
    def __init__(self, observer: Observer):
        self.observer = observer

    def run(self, start: TreeIterator) -> None:
        stack = [StackEntry(start)]

        while len(stack) > 0:
            current = stack[-1]
            self.observer.observe(current.state)
            try:
                next_child = next(current.child_iterator)
                stack.append(StackEntry(next_child))
            except StopIteration:
                stack.pop()
