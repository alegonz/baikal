import pytest

from baikal._core.digraph import DiGraph, NodeNotFoundError, CyclicDiGraphError


def test_add_node():
    graph = DiGraph()
    graph.add_node("A")
    assert "A" in graph


def test_add_same_node_twice():
    graph = DiGraph()
    graph.add_node("A")
    graph.add_node("A")
    assert "A" in graph


def test_add_edge():
    graph = DiGraph()
    graph.add_node("A")
    graph.add_node("B")
    graph.add_edge("A", "B")
    assert "B" in graph.successors("A") and "A" in graph.predecessors("B")


def test_add_edge_with_nonexistent_node():
    graph = DiGraph()
    graph.add_node("A")
    with pytest.raises(NodeNotFoundError):
        graph.add_edge("A", "B")


def test_can_add_same_node():
    graph = DiGraph()
    graph.add_node("A")
    graph.add_node("A")


def test_can_add_same_edge():
    graph = DiGraph()
    graph.add_node("A")
    graph.add_node("B")
    graph.add_edge("A", "B")
    graph.add_edge("A", "B")


def test_get_edge_data():
    graph = DiGraph()
    graph.add_node("A")
    graph.add_node("B")

    graph.add_edge("A", "B")
    assert graph.get_edge_data("A", "B") == set()

    graph.add_edge("A", "B", 123)
    assert graph.get_edge_data("A", "B") == {123}

    graph.add_edge("A", "B", 456, 789)
    assert graph.get_edge_data("A", "B") == {123, 456, 789}


def test_edges():
    graph = DiGraph()
    graph.add_node("A")
    graph.add_node("B")
    graph.add_node("C")

    graph.add_edge("A", "B", 123)
    graph.add_edge("A", "C", 456)

    # Cannot make sets of sets to compare and assert
    # so we use lists and do brute-force comparison
    def equal(x, y):
        y = list(y)
        try:
            for elem in x:
                y.remove(elem)
        except ValueError:
            return False
        return not y

    assert equal([("A", "B", {123}), ("A", "C", {456})], list(graph.edges))


def test_in_degree():
    graph = DiGraph()
    nodes = range(4)
    for node in nodes:
        graph.add_node(node)

    graph.add_edge(0, 1)
    graph.add_edge(1, 3)
    graph.add_edge(2, 3)

    assert [graph.in_degree(node) for node in nodes] == [0, 1, 0, 2]


def test_ancestors():
    graph = DiGraph()

    with pytest.raises(NodeNotFoundError):
        graph.ancestors(0)

    # Case 1:
    #  +--> [1] --> [3] --> [5]
    #  |                     ^
    # [0]                    |
    #  |                     |
    #  +--> [2] --> [4] -----+
    for node in range(6):
        graph.add_node(node)

    graph.add_edge(0, 1)
    graph.add_edge(1, 3)
    graph.add_edge(3, 5)

    graph.add_edge(0, 2)
    graph.add_edge(2, 4)
    graph.add_edge(4, 5)

    assert graph.ancestors(0) == set()
    assert graph.ancestors(4) == {0, 2}
    assert graph.ancestors(5) == {0, 1, 2, 3, 4}

    # Case 2:
    #  [0] --> [1] --> [4]
    #           ^       ^
    #           |       |
    #   + ------+       |
    #   |               |
    #  [2] --> [3] -----+
    graph = DiGraph()
    for node in range(5):
        graph.add_node(node)

    graph.add_edge(0, 1)
    graph.add_edge(1, 4)

    graph.add_edge(2, 1)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)

    assert graph.ancestors(0) == set()
    assert graph.ancestors(1) == {0, 2}
    assert graph.ancestors(3) == {2}
    assert graph.ancestors(4) == {0, 1, 2, 3}


def test_topological_sort():
    # Example randomly generated with
    # https://www.cs.usfca.edu/~galles/visualization/TopoSortDFS.html
    graph = DiGraph()
    for node in range(8):
        graph.add_node(node)

    graph.add_edge(0, 2)
    graph.add_edge(0, 3)
    graph.add_edge(2, 4)
    graph.add_edge(2, 6)
    graph.add_edge(4, 7)
    graph.add_edge(6, 7)
    graph.add_edge(3, 5)
    graph.add_edge(1, 5)
    graph.add_edge(3, 7)

    assert graph.topological_sort() == [1, 0, 3, 5, 2, 6, 4, 7]


def test_topological_sort_empty_graph():
    graph = DiGraph()
    assert graph.topological_sort() == []


def test_topological_sort_single_node():
    graph = DiGraph()
    graph.add_node(0)
    assert graph.topological_sort() == [0]


def test_topological_sort_cyclic_graph():
    graph = DiGraph()
    for node in [0, 1, 2]:
        graph.add_node(node)

    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 0)

    with pytest.raises(CyclicDiGraphError):
        graph.topological_sort()


def test_node_ordering():
    graph = DiGraph()
    nodes = [10, 0, 20, 40, 30]
    for node in nodes:
        graph.add_node(node)

    assert list(graph) == nodes


def test_clear():
    graph = DiGraph()
    graph.add_node(0)
    graph.add_node(1)
    graph.add_edge(0, 1, "foo")

    assert 0 in graph
    assert 1 in graph
    assert [(0, 1, {"foo"})] == list(graph.edges)

    graph.clear()

    assert 0 not in graph
    assert 1 not in graph
    assert [] == list(graph)
    assert [] == list(graph.edges)
