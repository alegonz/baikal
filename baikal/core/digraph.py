from collections import deque

from baikal.core.utils import make_name, make_repr


class NodeNotFoundError(Exception):
    """Exception raised when attempting to operate on a node that
    does not exist in the graph.
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
        if from_node not in self:
            raise NodeNotFoundError('{} is not in the graph!'.format(from_node))
        if to_node not in self:
            raise NodeNotFoundError('{} is not in the graph!'.format(to_node))

        self._successors[from_node].add(to_node)
        self._predecessors[to_node].add(from_node)

    def __contains__(self, node):
        return node in self._successors

    def __iter__(self):
        return iter(self._successors)

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

    def in_degree(self, node):
        if node not in self:
            raise NodeNotFoundError('{} is not in the graph!'.format(node))

        return len(self._predecessors[node])

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
        # Use name as is if it was specified by the user, to avoid the user a surprise
        self.name = name if name is not None else self._generate_unique_name()

    def _generate_unique_name(self):
        name = self.__class__.__name__

        n_instances = self._names.get(name, 0)
        unique_name = make_name(name, n_instances, sep='_')

        n_instances += 1
        self._names[name] = n_instances

        return unique_name

    @classmethod
    def clear_names(cls):
        cls._names.clear()

    def __repr__(self):
        return make_repr(self, ['name'])
