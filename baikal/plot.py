import io
import os

import pydot

from baikal.core.model import Model
from baikal.core.step import InputStep
from baikal.core.utils import safezip2


def plot_model(model, filename=None, show=False, expand_nested=False):
    """Plot the model"""

    dot_graph = pydot.Dot(graph_type='digraph', dpi=300)
    edges_built = set()

    def dummy_node(name):
        return pydot.Node(name=name, shape='rect', color='white', fontcolor='white',
                          fixedsize=True, width=0.0, height=0.0, fontsize=0.0)

    def build_edge(from_step, to_step, label, container):
        edge_key = (from_step, to_step, label)
        if edge_key not in edges_built:
            container.add_edge(pydot.Edge(src=from_step.name, dst=to_step.name, label=label))
            edges_built.add(edge_key)

    def build_dot_from(output, container=None):
        if container is None:
            container = dot_graph

        parent_step = output.step

        if isinstance(parent_step, Model) and expand_nested:
            # Build nested model
            nested_model = parent_step
            cluster = pydot.Cluster(name=nested_model.name, label=nested_model.name, style='dashed')

            for output in nested_model._internal_outputs:
                build_dot_from(output, cluster)
            container.add_subgraph(cluster)

            # Connect with outer model
            for input, internal_input in safezip2(nested_model.inputs, nested_model._internal_inputs):
                build_edge(input.step, internal_input.step, input.name, container)

        else:
            # Build step
            if isinstance(parent_step, InputStep):
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

        # Continue building
        for input in parent_step.inputs:
            build_dot_from(input)

    for output in model._internal_outputs:
        build_dot_from(output)
        # draw outgoing edges from model outputs
        dot_graph.add_node(dummy_node(output.name))
        dot_graph.add_edge(pydot.Edge(src=output.step.name, dst=output.name, label=output.name))

    # save plot
    if filename:
        basename, ext = os.path.splitext(filename)
        with open(filename, 'wb') as fh:
            fh.write(dot_graph.create(format=ext.lstrip('.').lower()))

    # display graph via matplotlib
    if show:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        png = dot_graph.create(format='png')
        img = mpimg.imread(io.BytesIO(png))
        plt.imshow(img, aspect='equal')
        plt.axis('off')
        plt.show()
