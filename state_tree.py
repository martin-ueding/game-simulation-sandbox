import json

class Node(object):
    def __init__(self, state, parent):
        self.state = state
        #self.children = []
        #self.parent = parent

    def __repr__(self):
        return json.dumps(self.state, sort_keys=True)

class Tree(object):
    def __init__(self, state):
        self.root = Node(state, None)
        self.leaves = [self.root]

    def insert(self, parent, state):
        node = Node(state, parent)
        #parent.children.append(node)
        self.leaves.append(node)

    def reset_leaves(self):
        self.leaves = []