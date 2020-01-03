from collections import deque, defaultdict, OrderedDict


class NodeNotFoundError(Exception):
    """Exception raised when attempting to operate on a node that
    does not exist in the graph.
    """


class CyclicDiGraphError(Exception):
    """Exception raised when graph has cycles.
    """


class DiGraph:
    def __init__(self, name=None):
        # We represent the adjacency matrix as a dict of dicts:
        # key: source node -> value: (key: destination node -> value: edge data (set))
        # Also, the graph nodes are stored in the order they were added.
        self._successors = OrderedDict()
        self._predecessors = OrderedDict()
        self.name = name

    def add_node(self, node):
        if node in self:
            # If node already exists in the graph, return silently.
            return
        # Edge data is stored in dict value as a set.
        # Currently, it is used to store the DataPlaceholder(s) associated with the (multi)edge.
        self._successors[node] = defaultdict(set)
        self._predecessors[node] = defaultdict(set)

    def add_edge(self, from_node, to_node, *edge_data):
        self._check_node_in_graph(from_node)
        self._check_node_in_graph(to_node)
        self._successors[from_node][to_node].update(edge_data)
        self._predecessors[to_node][from_node].update(edge_data)

    def get_edge_data(self, from_node, to_node):
        self._check_node_in_graph(from_node)
        self._check_node_in_graph(to_node)
        return self._successors[from_node][to_node]

    def __contains__(self, node):
        return node in self._successors

    def __iter__(self):
        return iter(self._successors)

    def clear(self):
        self._successors.clear()
        self._predecessors.clear()

    @property
    def edges(self):
        for from_node in self._successors:
            for to_node in self._successors[from_node]:
                edge_data = self._successors[from_node][to_node]
                yield (from_node, to_node, edge_data)

    def successors(self, node):
        self._check_node_in_graph(node)
        return iter(self._successors[node])

    def predecessors(self, node):
        self._check_node_in_graph(node)
        return iter(self._predecessors[node])

    def ancestors(self, node):
        self._check_node_in_graph(node)

        ancestors = set()
        for predecessor in self._predecessors[node]:
            ancestors.add(predecessor)
            ancestors |= self.ancestors(predecessor)
        return ancestors

    def in_degree(self, node):
        self._check_node_in_graph(node)
        return len(self._predecessors[node])

    def _check_node_in_graph(self, node):
        if node not in self:
            raise NodeNotFoundError("{} is not in the graph.".format(node))

    def topological_sort(self):
        # Implemented using depth-first search
        # Also works as a test of acyclicity
        n_nodes = len(self._successors)

        sorted_nodes = deque(maxlen=n_nodes)
        visited_nodes = set()
        unvisited_nodes = deque(
            sorted(self._predecessors, key=lambda k: len(self._predecessors[k])),
            maxlen=n_nodes,
        )

        # It is not mandatory, but to have more intuitive orderings,
        # we start depth-first search from nodes without predecessors (inputs)

        def visit(node):
            if node in sorted_nodes:
                return
            if node in visited_nodes:
                raise CyclicDiGraphError("DiGraph is not acyclic.")

            visited_nodes.add(node)
            for successor in self._successors[node]:
                visit(successor)

            visited_nodes.remove(node)
            sorted_nodes.appendleft(node)

        while unvisited_nodes:
            next_node = unvisited_nodes.popleft()
            visit(next_node)

        return list(sorted_nodes)
