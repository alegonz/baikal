__all__ = ["plot_model"]

import io
import os
from typing import Optional

try:
    import pydot
except ImportError:  # pragma: no cover
    raise ImportError(
        "Could not import pydot package."
        "You can install with `pip install pydot` or"
        "`pip install baikal[viz]`"
    )

from baikal import Model
from baikal._core.utils import make_name, safezip2


def _is_model(node):
    return isinstance(node.step, Model)


def _is_input(node):
    return not node.inputs


def quoted(s):
    return '"{}"'.format(s)


def dot_node(name, label):
    return pydot.Node(name=quoted(name), label=quoted(label), shape="rect")


def dot_input_node(name, label):
    return pydot.Node(
        name=quoted(name), label=quoted(label), shape="invhouse", color="green"
    )


def dot_edge(src, dst, label, color):
    return pydot.Edge(
        src=quoted(src), dst=quoted(dst), label=quoted(label), color=color
    )


def dot_cluster(name, label):
    return pydot.Cluster(graph_name=quoted(name), label=quoted(label), style="dashed")


def dummy_dot_node(name):
    return pydot.Node(
        name=quoted(name),
        shape="rect",
        color="white",
        fontcolor="white",
        fixedsize=True,
        width=0.0,
        height=0.0,
        fontsize=0.0,
    )


class _DotTransformer:
    def __init__(self, expand_nested, **dot_kwargs):
        self.expand_nested = expand_nested
        self.dot_kwargs = dot_kwargs
        self.node_names = {}
        self.inner_dot_nodes = {}

    def transform(self, model, outer_port=0, container=None, level=None):
        """Transform model graph to a dot graph. It will transform nested sub-models
        recursively, in which case it returns the dot nodes of the sub-model where the
        enclosing model should connect the edges of the steps that precede and follow
        the sub-model.
        """
        container = (
            pydot.Dot(graph_type="digraph", **self.dot_kwargs)
            if container is None
            else container
        )
        level = 0 if level is None else level
        root_name = make_name(model.name, outer_port, sep=":")

        # Add nodes
        for node in model.graph:
            if _is_input(node):
                name = make_name(root_name, node.step.name)
                label = node.step.name
                container.add_node(dot_input_node(name, label))

            elif _is_model(node) and self.expand_nested:
                name = make_name(root_name, node.name)
                label = node.name
                cluster = dot_cluster(name, label)
                container.add_subgraph(cluster)
                self.inner_dot_nodes[outer_port, node] = self.transform(
                    node.step, node.port, cluster, level + 1
                )

            else:
                name = make_name(root_name, node.name)
                label = node.name
                container.add_node(dot_node(name, label))

            self.node_names[outer_port, node] = name

        # Add edges
        for parent_node, node, dataplaceholders in model.graph.edges:
            for d in dataplaceholders:
                color = "orange" if d in node.targets else "black"

                if (_is_model(parent_node) or _is_model(node)) and self.expand_nested:

                    if _is_model(parent_node):
                        output_srcs, *_ = self.inner_dot_nodes[outer_port, parent_node]
                        src = output_srcs[parent_node.outputs.index(d)]
                    else:
                        src = self.node_names[outer_port, parent_node]

                    if _is_model(node):
                        if d in node.targets:
                            *_, target_dsts = self.inner_dot_nodes[outer_port, node]
                            dst = target_dsts[node.targets.index(d)]
                        else:
                            _, input_dsts, _ = self.inner_dot_nodes[outer_port, node]
                            dst = input_dsts[node.inputs.index(d)]
                    else:
                        dst = self.node_names[outer_port, node]

                else:
                    # Not expanded case, or step -> step case
                    color = "orange" if d in node.targets else "black"
                    src = self.node_names[outer_port, parent_node]
                    dst = self.node_names[outer_port, node]

                label = d.name
                container.add_edge(dot_edge(src, dst, label, color))

        if self.expand_nested and level > 0:
            return self.get_internal_dot_nodes(model, outer_port)

        else:
            self.build_output_edges(model, outer_port, container)

        return container

    def get_internal_dot_nodes(self, model, outer_port):
        """Get the dot nodes of the submodel where the enclosing model should
        connect the edges of the steps that precede and follow the submodel.
        """
        dot_input_dst = []
        for input in model._internal_inputs:
            dst = self.node_names[outer_port, input.node]
            dot_input_dst.append(dst)
        dot_target_dst = []
        for target in model._internal_targets:
            dst = self.node_names[outer_port, target.node]
            dot_target_dst.append(dst)
        dot_output_src = []
        for output in model._internal_outputs:
            src = self.node_names[outer_port, output.node]
            dot_output_src.append(src)
        return dot_output_src, dot_input_dst, dot_target_dst

    def build_output_edges(self, model, outer_port, container):
        root_name = make_name(model.name, outer_port, sep=":")
        keys = self.get_innermost_outputs_keys(model, outer_port)
        for (outer_port, inner_output), output in safezip2(
            keys, model._internal_outputs
        ):
            src = self.node_names[outer_port, inner_output.node]
            dst = make_name(root_name, output.name)
            label = output.name
            container.add_node(dummy_dot_node(dst))
            container.add_edge(dot_edge(src, dst, label, "black"))

    def get_innermost_outputs_keys(self, model, outer_port):
        """Get (outer_port, node) keys of the output placeholders at the innermost level.

        When the final step of a model is a sub-model and expand_nested is True, we plot
        the sub-models internal outputs as the model outputs. It the sub-model itself
        contains a sub-model, we continue recursively until hitting a non-Model step.
        """
        keys = []
        for output in model._internal_outputs:
            if self.expand_nested and _is_model(output.node):
                keys.extend(
                    self.get_innermost_outputs_keys(output.node.step, output.port)
                )
            else:
                keys.append((outer_port, output))
        return keys


# TODO: Add option to not plot targets
def plot_model(
    model: Model,
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
    ::

        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from baikal.plot import plot_model

        dot_graph = plot_model(model, include_targets=True, expand_nested=False)
        png = dot_graph.create(format="png", prog=prog)
        img = mpimg.imread(io.BytesIO(png))
        plt.imshow(img, aspect="equal")
        plt.axis("off")
        plt.show()

    """

    dot_transformer = _DotTransformer(expand_nested, **dot_kwargs)
    dot_graph = dot_transformer.transform(model)

    # save plot
    if filename:
        basename, ext = os.path.splitext(filename)
        with open(filename, "wb") as fh:
            fh.write(dot_graph.create(format=ext.lstrip(".").lower(), prog=prog))

    # display graph via matplotlib
    if show:  # pragma: no cover
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        png = dot_graph.create(format="png", prog=prog)
        img = mpimg.imread(io.BytesIO(png))
        plt.imshow(img, aspect="equal")
        plt.axis("off")
        plt.show()

    return dot_graph
