import io
import os
from typing import Set, Tuple, Optional

try:
    import pydot
except ImportError:
    raise ImportError(
        "Could not import pydot package."
        "You can install with `pip install pydot` or"
        "`pip install baikal[viz]`"
    )

from baikal._core.model import Model, Step
from baikal._core.utils import safezip2


# TODO: Add option to plot targets
def plot_model(
    model: Model,
    filename: Optional[str] = None,
    show: bool = False,
    include_targets: bool = True,
    expand_nested: bool = False,
    prog: str = "dot",
    **dot_kwargs
):
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

    include_targets
        Whether to include the model targets or not.

    expand_nested
        Whether to expand any nested models or not (display the nested model as
        a single step).

    prog
        Program to use to process the dot file into a graph.

    dot_kwargs
        Keyword arguments to pydot.Dot.
    """

    dot_graph = pydot.Dot(graph_type="digraph", **dot_kwargs)
    nodes_built = set()  # type: Set[Step]
    edges_built = set()  # type: Set[Tuple[Step, Step, str]]

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

    def get_internal_output(model_output):
        model = model_output.step
        index = model.outputs.index(model_output)
        internal_output = model._internal_outputs[index]
        return internal_output

    def build_step(step, is_input_or_target, container):
        if is_input_or_target:
            container.add_node(
                pydot.Node(name=step.name, shape="invhouse", color="green")
            )
        else:
            container.add_node(pydot.Node(name=step.name, shape="rect"))

        # Build incoming edges
        color = "black"
        for input in step.inputs:
            if isinstance(input.step, Model) and expand_nested:
                internal_output = get_internal_output(input)
                build_edge(internal_output.step, step, input.name, color, container)
            else:
                build_edge(input.step, step, input.name, color, container)

        if include_targets:
            color = "orange"
            for target in step.targets:
                if isinstance(target.step, Model) and expand_nested:
                    internal_output = get_internal_output(target)
                    build_edge(
                        internal_output.step, step, target.name, color, container
                    )
                else:
                    build_edge(target.step, step, target.name, color, container)

    def build_edge(from_step, to_step, label, color, container):
        edge_key = (from_step, to_step, label)

        if to_step is None:
            container.add_node(dummy_node(label))
            dst = label
        else:
            dst = to_step.name

        if edge_key not in edges_built:
            edge = pydot.Edge(src=from_step.name, dst=dst, label=label, color=color)
            container.add_edge(edge)
            edges_built.add(edge_key)

    def build_nested_model(model_, container):
        cluster = pydot.Cluster(
            graph_name=model_.name, label=model_.name, style="dashed"
        )

        for internal_output in model_._internal_outputs:
            build_dot_from(model_, internal_output, cluster)

        container.add_subgraph(cluster)

        # Connect with outer model
        color = "black"
        for input, _input in safezip2(model_.inputs, model_._internal_inputs):
            if isinstance(input.step, Model) and expand_nested:
                internal_output = get_internal_output(input)
                build_edge(
                    internal_output.step, _input.step, input.name, color, container
                )
            else:
                build_edge(input.step, _input.step, input.name, color, container)

        if include_targets:
            color = "orange"
            for target, _target in safezip2(model_.targets, model_._internal_targets):
                if isinstance(target.step, Model) and expand_nested:
                    internal_output = get_internal_output(target)
                    build_edge(
                        internal_output.step,
                        _target.step,
                        target.name,
                        color,
                        container,
                    )
                else:
                    build_edge(target.step, _target.step, target.name, color, container)

    def build_dot_from(model_, output_, container=None):
        if container is None:
            container = dot_graph

        parent_step = output_.step

        if parent_step in nodes_built:
            return

        if isinstance(parent_step, Model) and expand_nested:
            build_nested_model(parent_step, container)
        else:
            is_input_or_target = parent_step in [
                _dp.step for _dp in model_._internal_inputs + model_._internal_targets
            ]
            build_step(parent_step, is_input_or_target, container)

        nodes_built.add(parent_step)

        # Continue building
        for input in parent_step.inputs:
            build_dot_from(model_, input)

        if include_targets:
            for target in parent_step.targets:
                build_dot_from(model_, target)

    for output in model._internal_outputs:
        build_dot_from(model, output)
        color = "black"
        if isinstance(output.step, Model) and expand_nested:
            internal_output = get_internal_output(output)
            build_edge(internal_output.step, None, output.name, color, dot_graph)
        else:
            build_edge(output.step, None, output.name, color, dot_graph)

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
