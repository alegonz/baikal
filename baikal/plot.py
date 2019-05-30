import io
import os
from typing import Optional

try:
    import pydot
except ImportError:
    raise ImportError('Could not import pydot package.'
                      'You can install with `pip install pydot` or'
                      '`pip install baikal[viz]`')

from baikal._core.model import Model
from baikal._core.utils import safezip2


def plot_model(model: Model,
               filename: Optional[str] = None,
               show: bool = False,
               expand_nested: bool = False,
               prog: str = 'dot',
               **dot_kwargs):
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
    """

    dot_graph = pydot.Dot(graph_type='digraph', **dot_kwargs)
    nodes_built = set()
    edges_built = set()

    def dummy_node(name):
        return pydot.Node(name=name, shape='rect', color='white', fontcolor='white',
                          fixedsize=True, width=0.0, height=0.0, fontsize=0.0)

    def build_edge(from_step, to_step, label, container):
        edge_key = (from_step, to_step, label)
        if edge_key not in edges_built:
            container.add_edge(pydot.Edge(src=from_step.name, dst=to_step.name, label=label))
            edges_built.add(edge_key)

    def build_dot_from(model_, output_, container=None):
        if container is None:
            container = dot_graph

        parent_step = output_.step

        if parent_step in nodes_built:
            return

        if isinstance(parent_step, Model) and expand_nested:
            # Build nested model
            nested_model = parent_step
            cluster = pydot.Cluster(name=nested_model.name, label=nested_model.name, style='dashed')

            for output_ in nested_model._internal_outputs:
                build_dot_from(nested_model, output_, cluster)
            container.add_subgraph(cluster)

            # Connect with outer model
            for input, internal_input in safezip2(nested_model.inputs, nested_model._internal_inputs):
                build_edge(input.step, internal_input.step, input.name, container)

        else:
            # Build step
            if parent_step in [input.step for input in model_._internal_inputs]:
                container.add_node(pydot.Node(name=parent_step.name, shape='invhouse', color='green'))
            else:
                container.add_node(pydot.Node(name=parent_step.name, shape='rect'))

            # Build incoming edges
            for input in parent_step.inputs:
                if isinstance(input.step, Model) and expand_nested:
                    nested_model = input.step
                    index = nested_model.outputs.index(input)
                    internal_output = nested_model._internal_outputs[index]
                    build_edge(internal_output.step, parent_step, input.name, container)
                else:
                    build_edge(input.step, parent_step, input.name, container)

        nodes_built.add(parent_step)

        # Continue building
        for input in parent_step.inputs:
            build_dot_from(model_, input)

    for output in model._internal_outputs:
        build_dot_from(model, output)
        # draw outgoing edges from model outputs
        dot_graph.add_node(dummy_node(output.name))
        dot_graph.add_edge(pydot.Edge(src=output.step.name, dst=output.name, label=output.name))

    # save plot
    if filename:
        basename, ext = os.path.splitext(filename)
        with open(filename, 'wb') as fh:
            fh.write(dot_graph.create(format=ext.lstrip('.').lower(), prog=prog))

    # display graph via matplotlib
    if show:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        png = dot_graph.create(format='png', prog=prog)
        img = mpimg.imread(io.BytesIO(png))
        plt.imshow(img, aspect='equal')
        plt.axis('off')
        plt.show()
