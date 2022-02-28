from typing import Any
from typing import Dict
from typing import Generator

from .interface import Observer
from .interface import TreeIterator


class MockObserver(Observer):
    def __init__(self):
        self.states = []

    def observe(self, state: TreeIterator) -> None:
        self.states.append(state)


class DictTreeIterator(TreeIterator):
    def __init__(self, name: str, children: Dict[str, Any]):
        self.name = name
        self.children = children

    def is_terminal(self) -> bool:
        return len(self.children) > 0

    def get_children(self) -> Generator["DictTreeIterator", None, None]:
        for key, value in sorted(self.children.items()):
            yield DictTreeIterator(key, value)

    def name(self) -> str:
        return self.name
