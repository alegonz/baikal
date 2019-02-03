from typing import Union, List


def make_name(*parts, sep='/'):
    return sep.join([str(p) for p in parts])


class DiGraph:
    def __init__(self, name=None):
        self._adjacency = dict()
        self.name = name

    def add_node(self, node):
        self._adjacency[node] = set()  # successors (TODO: perhaps should predecessors)

    def add_edge(self, from_node, to_node):
        self._adjacency[from_node].add(to_node)

    def __contains__(self, node):
        return node in self._adjacency

    def __iter__(self):
        return iter(self._adjacency)

    def clear(self):
        self._adjacency.clear()

    @property
    def nodes(self):
        return list(self._adjacency.keys())

    # TODO: define topological_sort method
    # TODO: maybe should have a check for acyclicity


default_graph = DiGraph(name='default')


class Data:
    def __init__(self, shape, node, index=0):
        self.shape = shape
        self.node = node
        self.index = index
        self.name = make_name(node.name, index)


# TODO: '/0' at the end is cumbersome and unnecessary in InputNode's
# TODO: Consider removing the graph name from the node name
class Node:
    _names = dict()

    def __init__(self, name=None):
        # Maybe graph should be passed as a keyword argument
        self.graph = default_graph
        self.graph.add_node(self)

        if name is None:
            name = self.__class__.__name__

        name = make_name(self.graph.name, name)

        n_instances = self._names.get(name, 0)
        self.name = '_'.join([name, str(n_instances)])

        n_instances += 1
        self._names[name] = n_instances


# TODO: maybe should rename Processor to something more expressive: Component? Operation? Transformer?
class ProcessorMixin(Node):
    def __init__(self, name=None):
        super().__init__(name)
        self.outputs = None

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        if any([not isinstance(i, Data) for i in inputs]):
            raise ValueError('Processors must be called with Data inputs.')

        # Add edges
        for input in inputs:
            predecessor = input.node
            self.graph.add_edge(predecessor, self)

        self.outputs = self.build_outputs(inputs)
        return self.outputs

    # TODO: We might need a check_inputs method as well (Concatenate, Split, Merge, etc will need it).
    # Also, sklearn-based Processors can accept only shapes of length 1
    # (ignoring the samples, the dimensionality of the feature vector)

    def build_outputs(self, inputs: List[Data]) -> Union[Data, List[Data]]:
        # TODO: Maybe should be build_output_shapes.
        # It is not necessary to make the user import and use Data when extending this abstract mixin
        raise NotImplementedError


class InputNode(Node):
    def __init__(self, shape, name=None):
        super().__init__(name)
        self.outputs = Data(shape, self)


def Input(shape, name=None):
    # Maybe this can be implemented in InputNode.__new__
    input = InputNode(shape, name)
    return input.outputs

