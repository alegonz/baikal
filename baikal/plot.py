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

from baikal._core.model import Model as _Model
from baikal._core.step import InputStep as _InputStep
from baikal._core.utils import make_name as _make_name


def _is_model(step):
    return isinstance(step, _Model)


def _is_input(step):
    return isinstance(step, _InputStep)


class _DotTransformer:
    def __init__(self, expand_nested, **dot_kwargs):
        self.expand_nested = expand_nested
        self.dot_kwargs = dot_kwargs
        self.node_names = {}
        self.sub_internal_dot_nodes = {}

    def transform(self, model, container=None, level=None):
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
        root_name = model.name

        # Add nodes
        for node in model.graph:
            if _is_input(node.step):
                name = _make_name(root_name, node.step.name)
                label = node.step.name
                dot_node = pydot.Node(
                    name=name, label=label, shape="invhouse", color="green"
                )
                container.add_node(dot_node)

            elif _is_model(node.step) and self.expand_nested:
                name = _make_name(root_name, node.name)
                label = node.name
                cluster = pydot.Cluster(graph_name=name, label=label, style="dashed")
                container.add_subgraph(cluster)
                self.sub_internal_dot_nodes[node] = self.transform(
                    node.step, cluster, level + 1
                )

            else:
                name = _make_name(root_name, node.name)
                label = node.name
                dot_node = pydot.Node(name=name, label=label, shape="rect")
                container.add_node(dot_node)

            self.node_names[node] = name

        # Add edges
        for parent_node, node, dataplaceholders in model.graph.edges:
            for d in dataplaceholders:
                color = "orange" if d in node.targets else "black"

                if (
                    _is_model(parent_node.step) or _is_model(node.step)
                ) and self.expand_nested:

                    if _is_model(parent_node.step) and not _is_model(node.step):
                        # Case 1: submodel -> step
                        output_srcs, _, _ = self.sub_internal_dot_nodes[parent_node]
                        src, label = output_srcs[parent_node.outputs.index(d)]
                        dst = self.node_names[node]

                    elif not _is_model(parent_node.step) and _is_model(node.step):
                        # Case 2: step -> submodel
                        src = self.node_names[parent_node]

                        if d in node.targets:
                            _, _, target_dsts = self.sub_internal_dot_nodes[node]
                            dst = target_dsts[node.targets.index(d)]
                        else:
                            _, input_dsts, _ = self.sub_internal_dot_nodes[node]
                            dst = input_dsts[node.inputs.index(d)]
                        label = d.name

                    else:
                        # Case 3: submodel -> submodel
                        output_srcs, _, _ = self.sub_internal_dot_nodes[parent_node]
                        src, label = output_srcs[parent_node.outputs.index(d)]

                        if d in node.targets:
                            _, _, target_dsts = self.sub_internal_dot_nodes[node]
                            dst = target_dsts[node.targets.index(d)]
                        else:
                            _, input_dsts, _ = self.sub_internal_dot_nodes[node]
                            dst = input_dsts[node.inputs.index(d)]

                else:
                    # Not expanded case, or step -> step case
                    color = "orange" if d in node.targets else "black"
                    src = self.node_names[parent_node]
                    dst = self.node_names[node]
                    label = d.name

                dot_edge = pydot.Edge(src=src, dst=dst, label=label, color=color)
                container.add_edge(dot_edge)

        if self.expand_nested and level > 0:
            return self.get_internal_dot_nodes(model)

        else:
            self.build_output_edges(model, container)

        return container

    def get_internal_dot_nodes(self, model):
        """Get the dot nodes of the submodel where the enclosing model should
        connect the edges of the steps that precede and follow the submodel.
        """
        dot_input_dst = []
        for input in model._internal_inputs:
            dst = self.node_names[input.node]
            dot_input_dst.append(dst)
        dot_target_dst = []
        for target in model._internal_targets:
            dst = self.node_names[target.node]
            dot_target_dst.append(dst)
        dot_output_src = []
        for output in model._internal_outputs:
            src = self.node_names[output.node]
            label = output.name
            dot_output_src.append((src, label))
        return dot_output_src, dot_input_dst, dot_target_dst

    def build_output_edges(self, model, container):
        root_name = model.name
        outputs = self.get_innermost_outputs(model)
        for output in outputs:
            src = self.node_names[output.node]
            dst = _make_name(root_name, output.name)
            label = output.name
            container.add_node(self.dummy_dot_node(dst))
            container.add_edge(pydot.Edge(src=src, dst=dst, label=label, color="black"))

    def get_innermost_outputs(self, model):
        """Get the output placeholders at the innermost level.

        When the final step of a model is a sub-model and expand_nested is True, we plot
        the sub-models internal outputs as the model outputs. It the sub-model itself
        contains a sub-model, we continue recursively until hitting a non-Model step.
        """
        outputs = []
        for output in model._internal_outputs:
            if self.expand_nested and _is_model(output.node.step):
                outputs.extend(self.get_innermost_outputs(output.node.step))
            else:
                outputs.append(output)
        return outputs

    @staticmethod
    def dummy_dot_node(name):
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

    dot_transformer = _DotTransformer(expand_nested, **dot_kwargs)
    dot_graph = dot_transformer.transform(model)

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
