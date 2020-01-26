import pytest

from baikal import Model, Input
from baikal.plot import plot_model

from tests.helpers.fixtures import teardown
from tests.helpers.dummy_steps import DummyMIMO, DummySIMO, DummySISO, DummyMISO
from tests.helpers.sklearn_steps import LogisticRegression

PNG_FILENAME = "test_plot_model.png"


def build_submodel(step, number):
    x = Input(name="x_sub{}".format(number))
    y_t = Input(name="y_t_sub{}".format(number))
    y_p = step(x, y_t)
    return Model(x, y_p, y_t)


@pytest.mark.parametrize("expand_nested", [True, False])
def test_plot_model(teardown, tmp_path, expand_nested):
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

    model = Model(x, w, y_t, name="main_model")

    filename = str(tmp_path / "test_plot_model.png")
    plot_model(model, filename, show=False, expand_nested=expand_nested)


@pytest.mark.parametrize("expand_nested", [True, False])
def test_plot_independent_submodels(teardown, tmp_path, expand_nested):
    xs, y_ts, y_ps, steps = [], [], [], []

    for i in range(3):
        step = (
            LogisticRegression() if i == 0 else build_submodel(LogisticRegression(), i)
        )
        x = Input(name="x{}".format(i))
        y_t = Input(name="y_t{}".format(i))
        y_p = step(x, y_t)
        xs.append(x)
        y_ts.append(y_t)
        y_ps.append(y_p)

    model = Model(xs, y_ps, y_ts)

    filename = str(tmp_path / "test_plot_model.png")
    plot_model(model, filename, show=False, expand_nested=expand_nested)


@pytest.mark.parametrize("expand_nested", [True, False])
def test_plot_shared_submodel(teardown, tmp_path, expand_nested):
    xs, y_ts, y_ps = [], [], []
    submodel = build_submodel(LogisticRegression(), 0)

    for i in range(2):
        x = Input(name="x{}".format(i))
        y_t = Input(name="y_t{}".format(i))
        y_p = submodel(x, y_t)
        xs.append(x)
        y_ts.append(y_t)
        y_ps.append(y_p)

    model = Model(xs, y_ps, y_ts)

    filename = str(tmp_path / PNG_FILENAME)
    plot_model(model, filename, show=False, expand_nested=expand_nested)


@pytest.mark.parametrize("levels", [0, 1, 2, 3])
@pytest.mark.parametrize("expand_nested", [True, False])
def test_plot_nested_submodels(teardown, tmp_path, levels, expand_nested):

    submodels = [LogisticRegression()]
    for level in range(levels):
        sub_model = build_submodel(submodels[level], level + 1)
        submodels.append(sub_model)

    x = Input(name="x")
    y_t = Input(name="y_t")
    y_p = submodels[-1](x, y_t)
    model = Model(x, y_p, y_t)

    filename = str(tmp_path / "test_plot_model.png")
    plot_model(model, filename, show=False, expand_nested=expand_nested)
