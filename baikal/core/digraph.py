from baikal.core.utils import make_name


class DiGraph:
    def __init__(self, name=None):
        self._adjacency = dict()
        self.name = name

    def add_node(self, node):
        self._adjacency[node] = set()  # successors (TODO: perhaps should be predecessors)

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
    # used to keep track of number of instances and make unique names
    # a dict-of-dicts with graph and name as keys.
    _names = dict()

    def __init__(self, *args, name=None, **kwargs):
        super(Node, self).__init__(*args, **kwargs)
        # Maybe graph should be passed as a keyword argument
        graph = default_graph
        self.graph = graph
        self.graph.add_node(self)

        if graph not in self._names:
            self._names[graph] = {}

        self.name = self._generate_unique_name(name)

    def _generate_unique_name(self, name):
        graph = self.graph

        if name is None:
            name = self.__class__.__name__

        n_instances = self._names[graph].get(name, 0)
        unique_name = make_name(name, n_instances, sep='_')

        n_instances += 1
        self._names[graph][name] = n_instances

        return unique_name

    @classmethod
    def clear_names(cls):
        cls._names.clear()
