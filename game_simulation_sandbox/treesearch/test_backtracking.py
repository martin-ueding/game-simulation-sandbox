from .backtracking import Backtracking
from .mocks import DictTreeIterator
from .mocks import MockObserver


def test_backtracking() -> None:
    tree = {
        "Algorithms": {
            "Breadth-First": {},
            "Depth-First": {"Backtracking": {}, "Beam Search": {}},
            "Other": {"Monte Carlo Tree Search": {}},
        }
    }

    iterator = DictTreeIterator("Root", tree)
    observer = MockObserver()
    tree_search = Backtracking(observer)
    tree_search.run(iterator)

    assert [it.name for it in observer.states] == [
        "Root",
        "Algorithms",
        "Breadth-First",
        "Algorithms",
        "Depth-First",
        "Backtracking",
        "Depth-First",
        "Beam Search",
        "Depth-First",
        "Algorithms",
        "Other",
        "Monte Carlo Tree Search",
        "Other",
        "Algorithms",
        "Root",
    ]
