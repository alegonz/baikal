import pytest
import numpy as np
import sklearn.linear_model.logistic
from sklearn import datasets

from baikal.core.data import Data
from baikal.core.digraph import default_graph, Node, DiGraph
from baikal.core.step import Input, Step
from baikal.core.model import Model


@pytest.fixture
def teardown():
    yield
    Node.clear_names()
    default_graph.clear()


class TestInput:
    def test_instantiation(self, teardown):
        x0 = Input((10,))  # a 10-dimensional feature vector

        assert isinstance(x0, Data)
        assert (10,) == x0.shape
        assert 'InputNode_0/0' == x0.name

    def test_instantiate_two_with_same_name(self, teardown):
        x0 = Input((5,), name='x')
        x1 = Input((2,), name='x')

        assert 'x_0/0' == x0.name
        assert 'x_1/0' == x1.name

    def test_instantiate_two_without_name(self, teardown):
        x0 = Input((5,))
        x1 = Input((2,))

        assert 'InputNode_0/0' == x0.name
        assert 'InputNode_1/0' == x1.name


@pytest.fixture
def extended_sklearn_class():
    class LogisticRegression(Step, sklearn.linear_model.logistic.LogisticRegression):
        def __init__(self, *args, name=None, **kwargs):
            super(LogisticRegression, self).__init__(*args, name=name, **kwargs)

        def build_output_shapes(self, input_shapes):
            return [(1,)]

        def compute(self, x):
            return self.predict(x)

    return LogisticRegression


class TestStep:

    def test_call(self, extended_sklearn_class, teardown):
        x = Input((10,), name='x')
        y = extended_sklearn_class()(x)

        assert isinstance(y, Data)
        assert (1,) == y.shape
        assert 'LogisticRegression_0/0' == y.name

    def test_call_with_two_inputs(self, teardown):
        class MIMOStep(Step):
            def build_output_shapes(self, input_shapes):
                return [(1,), (1,)]

        x0 = Input((10,), name='x')
        x1 = Input((10,), name='x')
        y0, y1 = MIMOStep()([x0, x1])

        assert isinstance(y0, Data)
        assert isinstance(y1, Data)
        assert (1,) == y0.shape
        assert (1,) == y1.shape
        assert 'MIMOStep_0/0' == y0.name
        assert 'MIMOStep_0/1' == y1.name

    def test_instantiate_two_with_same_name(self, extended_sklearn_class, teardown):
        x = Input((10,), name='x')
        y0 = extended_sklearn_class(name='myclassifier')(x)
        y1 = extended_sklearn_class(name='myclassifier')(x)

        assert 'myclassifier_0/0' == y0.name
        assert 'myclassifier_1/0' == y1.name

    def test_instantiate_two_without_name(self, extended_sklearn_class, teardown):
        x = Input((10,), name='x')
        y0 = extended_sklearn_class()(x)
        y1 = extended_sklearn_class()(x)

        assert 'LogisticRegression_0/0' == y0.name
        assert 'LogisticRegression_1/0' == y1.name


class TestModel:
    def test_instantiation(self, extended_sklearn_class, teardown):
        x = Input((10,), name='x')
        y = extended_sklearn_class()(x)
        model = Model(x, y)

    def test_instantiation_with_wrong_input_type(self, extended_sklearn_class, teardown):
        x = Input((10,), name='x')
        y = extended_sklearn_class()(x)

        x_wrong = np.zeros((10,))
        with pytest.raises(ValueError):
            model = Model(x_wrong, y)

    def test_fit(self, extended_sklearn_class, teardown):
        # Based on the example in
        # https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
        iris = datasets.load_iris()

        X = iris.data[:, :2]  # we only take the first two features.
        y = iris.target

        x = Input((2,), name='x')
        yp = extended_sklearn_class(multi_class='multinomial')(x)

        model = Model(x, yp)
        model.fit(X, y)

        assert hasattr(extended_sklearn_class, 'coef_')


class TestDiGraph:
    def test_add_node(self):
        graph = DiGraph()
        graph.add_node('A')
        assert 'A' in graph.nodes

    def test_add_edge(self):
        graph = DiGraph()
        graph.add_node('A')
        graph.add_node('B')
        graph.add_edge('A', 'B')
        assert 'B' in graph.successors('A') and 'A' in graph.predecessors('B')

    def test_add_edge_with_nonexistent_node(self):
        graph = DiGraph()
        graph.add_node('A')
        with pytest.raises(KeyError):
            graph.add_edge('A', 'B')

    def test_topological_sort(self):
        # Example randomly generated with
        # https://www.cs.usfca.edu/~galles/visualization/TopoSortDFS.html
        graph = DiGraph()
        nodes = range(8)
        for node in nodes:
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
