import io
import os
from typing import Dict, Optional, Tuple

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


def build_dot_model(model, container, expand_nested, level):
    root_name = model.name
    node_names = {}  # type: Dict[_Node, str]
    sub_dot_nodes = {}  # type: Dict[_Node, Tuple]

    # Add nodes
    for node in model.graph:
        if isinstance(node.step, _InputStep):
            name = _make_name(root_name, node.step.name)
            label = node.step.name
            dot_node = pydot.Node(
                name=name, label=label, shape="invhouse", color="green"
            )
            container.add_node(dot_node)

        elif isinstance(node.step, _Model) and expand_nested:
            name = _make_name(root_name, node.name)
            label = node.name
            cluster = pydot.Cluster(graph_name=name, label=label, style="dashed")
            container.add_subgraph(cluster)
            sub_dot_nodes[node] = build_dot_model(
                node.step, cluster, expand_nested, level + 1
            )

        else:
            name = _make_name(root_name, node.name)
            label = node.name
            dot_node = pydot.Node(name=name, label=label, shape="rect")
            container.add_node(dot_node)

        node_names[node] = name

    # Add edges
    for parent_node, node, dataplaceholders in model.graph.edges:
        for d in dataplaceholders:
            color = "orange" if d in node.targets else "black"

            if expand_nested and (
                isinstance(parent_node.step, _Model) or isinstance(node.step, _Model)
            ):
                if isinstance(parent_node.step, _Model) and not isinstance(
                    node.step, _Model
                ):
                    # Case 1: submodel -> step
                    output_srcs, _, _ = sub_dot_nodes[parent_node]
                    src, label = output_srcs[parent_node.outputs.index(d)]
                    dst = node_names[node]

                elif not isinstance(parent_node.step, _Model) and isinstance(
                    node.step, _Model
                ):
                    # Case 2: step -> submodel
                    src = node_names[parent_node]

                    if d in node.targets:
                        _, _, target_dsts = sub_dot_nodes[node]
                        dst = target_dsts[node.targets.index(d)]
                    else:
                        _, input_dsts, _ = sub_dot_nodes[node]
                        dst = input_dsts[node.inputs.index(d)]
                    label = d.name

                else:
                    # Case 3: submodel -> submodel
                    output_srcs, _, _ = sub_dot_nodes[parent_node]
                    src, label = output_srcs[parent_node.outputs.index(d)]

                    if d in node.targets:
                        _, _, target_dsts = sub_dot_nodes[node]
                        dst = target_dsts[node.targets.index(d)]
                    else:
                        _, input_dsts, _ = sub_dot_nodes[node]
                        dst = input_dsts[node.inputs.index(d)]

            else:
                # Not expanded case, or step -> step case
                color = "orange" if d in node.targets else "black"
                src = node_names[parent_node]
                dst = node_names[node]
                label = d.name

            dot_edge = pydot.Edge(src=src, dst=dst, label=label, color=color)
            container.add_edge(dot_edge)

    if expand_nested and level > 0:
        # Return the dot nodes of the submodel where the enclosing model should
        # connect the edges of the steps that precede and follow the submodel
        dot_input_dst = []
        for input in model._internal_inputs:
            dst = node_names[input.node]
            dot_input_dst.append(dst)

        dot_target_dst = []
        for target in model._internal_targets:
            dst = node_names[target.node]
            dot_target_dst.append(dst)

        dot_output_src = []
        for output in model._internal_outputs:
            src = node_names[output.node]
            label = output.name
            dot_output_src.append((src, label))
        return dot_output_src, dot_input_dst, dot_target_dst

    else:
        # Add output edges of the root model
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

    build_dot_model(model, dot_graph, expand_nested, 0)

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
