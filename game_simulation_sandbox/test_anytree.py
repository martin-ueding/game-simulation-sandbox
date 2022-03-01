import anytree


def test_storage():
    root = anytree.AnyNode(id=1)
    assert len(root.children) == 0
    anytree.AnyNode(parent=root, id=1)
    assert len(root.children) == 1
