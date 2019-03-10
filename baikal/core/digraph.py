from collections import deque

from baikal.core.utils import make_name


class NodeNotFoundError(Exception):
    """Exception raised when attempting to operate on a node that
    does not exist in the graph.
    """


class MultiEdgeError(Exception):
    """Exception raised when attempting to add an edge that already exists.
    """


class CyclicDiGraphError(Exception):
    """Exception raised when graph has cycles.
    """


class DiGraph:
    def __init__(self, name=None):
        self._successors = dict()
        self._predecessors = dict()
        self.name = name

    def add_node(self, node):
        if node in self:
            # If node already exists in the graph, return silently.
            return
        self._successors[node] = set()
        self._predecessors[node] = set()

    def add_edge(self, from_node, to_node):
        if self.has_edge(from_node, to_node):
            raise MultiEdgeError(
                'An edge between {} and {} already exists (multiedges are not allowed)!'.format(from_node, to_node))

        self._successors[from_node].add(to_node)
        self._predecessors[to_node].add(from_node)

    def __contains__(self, node):
        return node in self._successors

    def __iter__(self):
        return iter(self._successors)

    def has_edge(self, from_node, to_node):
        if from_node not in self:
            raise NodeNotFoundError('{} is not in the graph!'.format(from_node))
        if to_node not in self:
            raise NodeNotFoundError('{} is not in the graph!'.format(to_node))

        return to_node in self._successors[from_node]

    def clear(self):
        self._successors.clear()
        self._predecessors.clear()

    @property
    def nodes(self):
        return list(self._successors)

    def successors(self, node):
        return self._successors[node]

    def predecessors(self, node):
        return self._predecessors[node]

    def ancestors(self, node):
        if node not in self:
            raise NodeNotFoundError('{} is not in the graph!'.format(node))

        ancestors = set()
        for predecessor in self._predecessors[node]:
            ancestors.add(predecessor)
            ancestors |= self.ancestors(predecessor)
        return ancestors

    def topological_sort(self):
        # Implemented using depth-first search
        # Also works as a test of acyclicity
        n_nodes = len(self._successors)

        sorted_nodes = deque(maxlen=n_nodes)
        visited_nodes = set()
        unvisited_nodes = deque(sorted(self._predecessors,
                                       key=lambda k: len(self._predecessors[k])),
                                maxlen=n_nodes)

        # It is not mandatory, but to have more intuitive orderings,
        # we start depth-first search from nodes without predecessors (inputs)

        def visit(node):
            if node in sorted_nodes:
                return
            if node in visited_nodes:
                raise CyclicDiGraphError('DiGraph is not acyclic!')

            visited_nodes.add(node)
            for successor in self._successors[node]:
                visit(successor)

            visited_nodes.remove(node)
            sorted_nodes.appendleft(node)

        while unvisited_nodes:
            node = unvisited_nodes.popleft()
            visit(node)

        return list(sorted_nodes)


default_graph = DiGraph(name='default')


class Node:
    # used to keep track of number of instances and make unique names
    # a dict-of-dicts with graph and name as keys.
    _names = dict()

    def __init__(self, *args, name=None, **kwargs):
        super(Node, self).__init__(*args, **kwargs)  # Necessary to use this class as a mixin
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
