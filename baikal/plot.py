import io
import os
from typing import Dict, Optional

try:
    import pydot
except ImportError:  # pragma: no cover
    raise ImportError(
        "Could not import pydot package."
        "You can install with `pip install pydot` or"
        "`pip install baikal[viz]`"
    )

from baikal._core.model import Model as _Model
from baikal._core.step import InputStep as _InputStep, Node as _Node
from baikal._core.utils import make_name as _make_name


def dummy_node(name):
    return pydot.Node(
        name=name,
        shape="rect",
        color="white",
        fontcolor="white",
        fixedsize=True,
        width=0.0,
        height=0.0,
        fontsize=0.0,
    )


def build_dot_model(model, container):
    root_name = model.name
    node_names = {}  # type: Dict[_Node, str]

    # TODO: handle nested model
    for node in model.graph:
        if isinstance(node.step, _InputStep):
            name = _make_name(root_name, node.step.name)
            label = node.step.name
            dot_node = pydot.Node(
                name=name, label=label, shape="invhouse", color="green"
            )
        else:
            name = _make_name(root_name, node.name)
            label = node.name
            dot_node = pydot.Node(name=name, label=label, shape="rect")
        container.add_node(dot_node)
        node_names[node] = name

    for from_node, to_node, dataplaceholders in model.graph.edges:
        for d in dataplaceholders:
            color = "orange" if d in to_node.targets else "black"
            src, dst = node_names[from_node], node_names[to_node]
            dot_edge = pydot.Edge(src=src, dst=dst, label=d.name, color=color)
            container.add_edge(dot_edge)

    for output in model._internal_outputs:
        src = node_names[output.node]
        dst = _make_name(root_name, output.name)
        label = output.name
        container.add_node(dummy_node(dst))
        container.add_edge(pydot.Edge(src=src, dst=dst, label=label, color="black"))


# TODO: Add option to not plot targets
def plot_model(
    model: _Model,
    filename: Optional[str] = None,
    show: bool = False,
    expand_nested: bool = False,
    prog: str = "dot",
    **dot_kwargs
) -> pydot.Dot:
    """Plot a model to file and/or display it.

    This function requires pydot and graphviz. It also requires matplotlib
    to display plots to the screen.

    Parameters
    ----------
    model
        The model to plot.

    filename
        Filename (optional).

    show
        Whether to display the plot in the screen or not. Requires matplotlib.

    expand_nested
        Whether to expand any nested models or not (display the nested model as
        a single step).

    prog
        Program to use to process the dot file into a graph.

    dot_kwargs
        Keyword arguments to pydot.Dot.

    Returns
    -------
    dot_graph
        Dot graph of the given model. It can be used to generate an image for plotting.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.image as mpimg
    >>> from baikal.plot import plot_model
    >>> dot_graph = plot_model(model, include_targets=True, expand_nested=False)
    >>> png = dot_graph.create(format="png", prog=prog)
    >>> img = mpimg.imread(io.BytesIO(png))
    >>> plt.imshow(img, aspect="equal")
    >>> plt.axis("off")
    >>> plt.show()
    """

    dot_graph = pydot.Dot(graph_type="digraph", **dot_kwargs)

    build_dot_model(model, dot_graph)

    # save plot
    if filename:
        basename, ext = os.path.splitext(filename)
        with open(filename, "wb") as fh:
            fh.write(dot_graph.create(format=ext.lstrip(".").lower(), prog=prog))

    # display graph via matplotlib
    if show:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        png = dot_graph.create(format="png", prog=prog)
        img = mpimg.imread(io.BytesIO(png))
        plt.imshow(img, aspect="equal")
        plt.axis("off")
        plt.show()

    return dot_graph


# if is_input_or_target:
#     node = pydot.Node(name=step.name, shape="invhouse", color="green")
# else:
#     node = pydot.Node(name=step.name, shape="rect")
#
# edge = pydot.Edge(src=src, dst=dst, label=label, color=color)
# cluster = pydot.Cluster(graph_name=name, label=name, style="dashed")
#
# container.add_node(n)
# container.add_edge(e)
# container.add_subgraph(cluster)
