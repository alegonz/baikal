import pytest

from baikal.core.digraph import DiGraph, NodeNotFoundError, CyclicDiGraphError


def test_add_node():
    graph = DiGraph()
    graph.add_node('A')
    assert 'A' in graph.nodes


def test_add_same_node_twice():
    graph = DiGraph()
    graph.add_node('A')
    graph.add_node('A')
    assert 'A' in graph.nodes


def test_add_edge():
    graph = DiGraph()
    graph.add_node('A')
    graph.add_node('B')
    graph.add_edge('A', 'B')
    assert 'B' in graph.successors('A') and 'A' in graph.predecessors('B')


def test_add_edge_with_nonexistent_node():
    graph = DiGraph()
    graph.add_node('A')
    with pytest.raises(NodeNotFoundError):
        graph.add_edge('A', 'B')


def test_can_add_same_node():
    graph = DiGraph()
    graph.add_node('A')
    graph.add_node('A')


def test_can_add_same_edge():
    graph = DiGraph()
    graph.add_node('A')
    graph.add_node('B')
    graph.add_edge('A', 'B')
    graph.add_edge('A', 'B')


def test_in_degree():
    graph = DiGraph()
    nodes = range(4)
    for node in nodes:
        graph.add_node(node)

    graph.add_edge(0, 1)
    graph.add_edge(1, 3)
    graph.add_edge(2, 3)

    assert [0, 1, 0, 2] == [graph.in_degree(node) for node in nodes]


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

    assert set() == graph.ancestors(0)
    assert {0, 2} == graph.ancestors(4)
    assert {0, 1, 2, 3, 4} == graph.ancestors(5)

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

    assert set() == graph.ancestors(0)
    assert {0, 2} == graph.ancestors(1)
    assert {2} == graph.ancestors(3)
    assert {0, 1, 2, 3} == graph.ancestors(4)


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

    assert [1, 0, 3, 5, 2, 6, 4, 7] == graph.topological_sort()


def test_topological_sort_empty_graph():
    graph = DiGraph()
    assert [] == graph.topological_sort()


def test_topological_sort_single_node():
    graph = DiGraph()
    graph.add_node(0)
    assert [0] == graph.topological_sort()


def test_topological_sort_cyclic_graph():
    graph = DiGraph()
    for node in [0, 1, 2]:
        graph.add_node(node)

    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 0)

    with pytest.raises(CyclicDiGraphError):
        graph.topological_sort()
