import pytest

from baikal import Model, Input
from baikal.plot import plot_model

from tests.helpers.fixtures import teardown
from tests.helpers.dummy_steps import DummyMIMO, DummySIMO, DummySISO, DummyMISO


@pytest.mark.parametrize("include_targets", [True, False])
@pytest.mark.parametrize("expand_nested", [True, False])
def test_plot_model(teardown, tmp_path, include_targets, expand_nested):
    # Below is a very contrived dummy model

    # ------- Sub-model 1
    x1_sub1 = Input(name="x1_sub1")
    x2_sub1 = Input(name="x2_sub1")
    y_t_sub1 = Input(name="y_t_sub1")
    y_p1_sub1, y_p2_sub1 = DummyMIMO()([x1_sub1, x2_sub1], y_t_sub1)
    submodel1 = Model(
        [x1_sub1, x2_sub1], [y_p1_sub1, y_p2_sub1], y_t_sub1, name="submodel1"
    )

    # ------- Sub-model 2
    y_t_sub2 = Input(name="y_t_sub2")
    y_p_sub2 = DummySISO()(y_t_sub2)
    submodel2 = Model(y_t_sub2, y_p_sub2, name="submodel2")

    # ------- Main model
    x = Input(name="x")
    y_t = Input(name="y_t")
    y_t_trans = submodel2(y_t)
    h1, h2 = DummySIMO()(x)
    z1, z2 = submodel1([h1, h2], y_t_trans)
    w = DummyMISO()([z1, z2])

    # a completely independent pipeline
    u = Input(name="u")
    v = DummySISO()(u)

    model = Model([x, u], [w, v], y_t, name="main_model")

    filename = str(tmp_path / "test_plot_model.png")
    plot_model(
        model,
        filename,
        show=False,
        include_targets=include_targets,
        expand_nested=expand_nested,
    )
