from typing import Generator


class TreeIterator:
    def is_terminal(self) -> bool:
        raise NotImplementedError()

    def get_children(self) -> Generator["TreeIterator", None, None]:
        raise NotImplementedError()


class TreeSearch:
    def run(self, start: TreeIterator) -> None:
        raise NotImplementedError()


class Observer:
    def observe(self, state: TreeIterator) -> None:
        raise NotImplementedError()
