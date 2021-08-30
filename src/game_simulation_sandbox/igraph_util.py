import pathlib
import subprocess


def render_igraph_neato(path: pathlib.Path) -> None:
    with open(path) as f:
        lines = list(f)
    lines.insert(2, "splines = true\n")
    lines.insert(2, "overlap = false\n")
    with open(path, "w") as f:
        for line in lines:
            f.write(line)

    subprocess.run(
        ["neato", "-T", "pdf", path.absolute(), "-o", path.with_suffix(".pdf")]
    )
    subprocess.run(
        ["neato", "-T", "svg", path.absolute(), "-o", path.with_suffix(".svg")]
    )
