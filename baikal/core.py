def make_name(*parts, sep='/'):
    return sep.join(parts)


class DiGraph:
    def __init__(self, name=None):
        self._adjacency = dict()
        self.name = name

    def add_node(self, node):
        self._adjacency[node] = set()  # successors

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

        # TODO: Need a way to determine the number of outputs
        # Usually is only one, but components like, e.g. Split
        # can have two or more outputs.
        self.outputs = Data(make_name(self.name, '0'), self)

        return self.outputs


class InputNode(Node):
    def __init__(self, name=None):
        super().__init__(name)
        self.output = Data(make_name(self.name, '0'), self)


def Input(name=None):
    input = InputNode(name)
    return input.output


class Data:
    def __init__(self, name, node):
        self.name = name
        self.node = node
