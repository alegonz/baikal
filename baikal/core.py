def make_name(*parts, sep='/'):
    return sep.join(parts)


class DiGraph:
    def __init__(self, name=None):
        self._adjacency = dict()
        self.name = name

    def add_node(self, node):
        self._adjacency[node] = set()  # successors

    # TODO: define add_edge method

    def __contains__(self, node):
        return node in self._adjacency

    def __iter__(self):
        return iter(self._adjacency)

    # TODO: define topological_sort method


default_graph = DiGraph(name='default')


class Data:
    def __init__(self, name):
        self.name = name


class Input:
    _names = dict()

    def __init__(self, name=None):
        graph = default_graph

        if name is None:
            name = self.__class__.__name__

        name = make_name(graph.name, name)

        n_instances = self._names.get(name, 0)
        self.name = '_'.join([name, str(n_instances)])

        n_instances += 1
        self._names[name] = n_instances

        self.output = Data(make_name(self.name, '0'))
        graph.add_node(self)


def create_input(name=None):
    input = Input(name)
    return input.output
