import pytest
import numpy as np
from sklearn import datasets
import sklearn.decomposition
import sklearn.linear_model.logistic
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

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
def sklearn_classifier_step():
    class LogisticRegression(Step, sklearn.linear_model.logistic.LogisticRegression):
        def __init__(self, name=None, **kwargs):
            super(LogisticRegression, self).__init__(name=name, **kwargs)

        def build_output_shapes(self, input_shapes):
            return [(1,)]

        def compute(self, x):
            return self.predict(x)

        @property
        def fitted(self):
            try:
                return check_is_fitted(self, ['coef_'], all_or_any=all)
            except NotFittedError:
                return False

    return LogisticRegression


@pytest.fixture
def sklearn_transformer_step():
    class PCA(Step, sklearn.decomposition.PCA):
        def __init__(self, name=None, **kwargs):
            super(PCA, self).__init__(name=name, **kwargs)

        def build_output_shapes(self, input_shapes):
            # TODO: How to handle when n_components is determined during fit?
            # This occurs when passed as a percentage of total variance (0 < n_components < 1)
            return [(self.n_components,)]

        def compute(self, x):
            return self.transform(x)

        @property
        def fitted(self):
            try:
                return check_is_fitted(self, ['mean_', 'components_'], all_or_any=all)
            except NotFittedError:
                return False

    return PCA


class TestStep:

    def test_call(self, sklearn_classifier_step, teardown):
        x = Input((10,), name='x')
        y = sklearn_classifier_step()(x)

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

    def test_instantiate_two_with_same_name(self, sklearn_classifier_step, teardown):
        x = Input((10,), name='x')
        y0 = sklearn_classifier_step(name='myclassifier')(x)
        y1 = sklearn_classifier_step(name='myclassifier')(x)

        assert 'myclassifier_0/0' == y0.name
        assert 'myclassifier_1/0' == y1.name

    def test_instantiate_two_without_name(self, sklearn_classifier_step, teardown):
        x = Input((10,), name='x')
        y0 = sklearn_classifier_step()(x)
        y1 = sklearn_classifier_step()(x)

        assert 'LogisticRegression_0/0' == y0.name
        assert 'LogisticRegression_1/0' == y1.name


class TestModel:
    def test_instantiation(self, sklearn_classifier_step, teardown):
        x = Input((10,), name='x')
        y = sklearn_classifier_step()(x)
        model = Model(x, y)

    def test_instantiation_with_wrong_input_type(self, sklearn_classifier_step, teardown):
        x = Input((10,), name='x')
        y = sklearn_classifier_step()(x)

        x_wrong = np.zeros((10,))
        with pytest.raises(ValueError):
            model = Model(x_wrong, y)

    def test_fit_classifier(self, sklearn_classifier_step, teardown):
        # Based on the example in
        # https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
        iris = datasets.load_iris()

        X_data = iris.data[:, :2]  # we only take the first two features.
        y_data = iris.target

        x = Input((2,), name='x')
        y = sklearn_classifier_step(multi_class='multinomial')(x)

        model = Model(x, y)

        # Style 1: pass data as in instantiation
        model.fit(X_data, y_data)
        assert y.node.fitted

        # FIXME: These will likely pass if the above passes.
        # Split into separate tests (init a new model everytime)

        # Style 2: pass a dict keyed by Data instances
        # model.fit({x: X_data, y: y_data})
        # assert y.node.fitted
        #
        # # Style 3: pass a dict keyed by Data instances names
        # model.fit({'x': X_data, 'LogisticRegression_0/0': y_data})
        # assert y.node.fitted

    def test_fit_transformer(self, sklearn_transformer_step, teardown):
        iris = datasets.load_iris()
        X_data = iris.data

        x = Input((4,), name='x')
        xt = sklearn_transformer_step(n_components=2)(x)

        model = Model(x, xt)
        model.fit(X_data)
        assert y.node.fitted


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

    def test_topological_sort_empty_graph(self):
        graph = DiGraph()
        assert [] == graph.topological_sort()

    def test_topological_sort_single_node(self):
        graph = DiGraph()
        graph.add_node(0)
        assert [0] == graph.topological_sort()

    def test_topological_sort_cyclic_graph(self):
        graph = DiGraph()
        for node in [0, 1, 2]:
            graph.add_node(node)

        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 0)

        with pytest.raises(ValueError):
            graph.topological_sort()
