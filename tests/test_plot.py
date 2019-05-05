from baikal.core.model import Model
from baikal.core.step import Input
from baikal.plot import plot_model

from tests.helpers.fixtures import teardown
from tests.helpers.dummy_steps import DummyMIMO, DummySIMO, DummySISO, DummyMISO


def test_plot_model(teardown, tmp_path):
    x1 = Input(name='x1')
    x2 = Input(name='x2')
    y1, y2 = DummyMIMO()([x1, x2])
    submodel = Model([x1, x2], [y1, y2], name='submodel')

    x = Input(name='x')
    h1, h2 = DummySIMO()(x)
    z1, z2 = submodel([h1, h2])

    u = Input(name='u')
    v = DummySISO()(u)

    w = DummyMISO()([z1, z2])
    model = Model([x, u], [w, v], name='main_model')

    filename = str(tmp_path / 'test_plot_model.png')
    plot_model(model, filename, show=False, expand_nested=True)
