import anytree.exporter
import click


@click.command()
def main():
    tree = anytree.Node(1)

    leaves = [tree]
    for level in range(20):
        new_leaves = []
        for leaf in leaves:
            new_leaves.append(anytree.Node(leaf.name * 2, parent=leaf))
            if leaf.name not in [1, 2, 4] and (leaf.name - 1) % 3 == 0:
                prev = (leaf.name - 1) // 3
                if prev % 2 == 1:
                    new_leaves.append(anytree.Node(prev, parent=leaf))
        leaves = new_leaves

    # print(anytree.RenderTree(tree))

    anytree.exporter.DotExporter(tree).to_dotfile("graph2.dot")


if __name__ == "__main__":
    main()
