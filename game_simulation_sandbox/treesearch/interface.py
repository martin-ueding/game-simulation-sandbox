from typing import Generator


class TreeIterator:
    def get_children(self) -> Generator["TreeIterator", None, None]:
        raise NotImplementedError()


class TreeSearch:
    def run(self, start: TreeIterator) -> None:
        raise NotImplementedError()


class Observer:
    def observe(self, state: TreeIterator) -> None:
        raise NotImplementedError()
