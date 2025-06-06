# tests/test_utils.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from magnetic_materials_2d.utils import (
    sorted_descriptors,
    top12,
    top_descriptors,
    best_descriptors,
    print_best_descriptors,
    important_descriptors,
    optimum_importance,
    print_loss,
    single_descriptor_regression,
    compare,
    parity_plot,
)


def make_multi_feature_df():
    """
    Create a DataFrame with two features:
      - f1 perfectly correlates with target (y = 2*x)
      - f2 is random noise
    """
    np.random.seed(0)
    f1 = np.linspace(0, 10, 20)
    f2 = np.random.randn(20)
    target = 2 * f1
    return pd.DataFrame({"f1": f1, "f2": f2, "target": target})


def test_sorted_descriptors_order():
    """sorted_descriptors should rank f1 above f2, since f1 perfectly predicts
    target."""
    df = make_multi_feature_df()
    model = LinearRegression()

    ranked = sorted_descriptors(df, target="target", model=model)
    assert isinstance(ranked, list)
    assert ranked[0][1] == "f1"
    assert ranked[1][1] == "f2"
    assert ranked[0][0] > ranked[1][0]


def test_top12_prints_twelve_lines(capsys):
    """
    top12 should print exactly 12 lines: 1 title + 1 underline + 12 descriptor lines.
    """
    desc_list = [(i / 10.0, f"col{i}") for i in range(20)]
    mapping = {f"col{i}": f"Column {i}" for i in range(20)}

    top12(desc_list, mapping, target="dummy_target", method="DummyModel")
    captured = capsys.readouterr().out.strip().splitlines()

    assert captured[0].startswith(
        "12 highest scored descriptors for dummy_target using DummyModel"
    )
    assert len(captured) == 2 + 12
    assert captured[2].startswith(" 1. Column 0")


def test_top_descriptors_logic():
    descriptors = [(0.9, "f1"), (0.5, "f2"), (0.1, "f3")]
    assert top_descriptors(descriptors, threshold=0.6) == ["f1"]
    assert top_descriptors(descriptors, threshold=0.2) == ["f1", "f2"]
    assert top_descriptors(descriptors, threshold=0.95) == []


def test_best_descriptors_handles_perfect_predictor(monkeypatch):
    """best_descriptors uses threshold=0.0 first (f1 & f2 both included =>
    R²=1.0), so it will return ['f1', 'f2'] rather than just ['f1']."""
    df = make_multi_feature_df()
    sorted_list = [(1.0, "f1"), (0.1, "f2")]
    model = LinearRegression()

    # Prevent plotting from hanging
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    best = best_descriptors(
        numeric_df=df,
        all_descriptors=sorted_list,
        model=model,
        target="target",
    )
    assert best == ["f1", "f2"]


def test_print_best_descriptors_outputs_correct_format(capsys):
    descriptors = ["f1", "f2", "f3"]
    mapping = {
        "f1": "Feature 1",
        "f2": "Feature 2",
        "f3": "Feature 3",
        "target": "Target",
    }

    print_best_descriptors(
        descriptors, mapping, target="target", method="LinReg"
    )
    out = capsys.readouterr().out.strip().splitlines()

    assert out[0] == "Best descriptors for Target using LinReg"
    assert set(out[1]) == set("-")
    assert "Feature 1 (f1)" in out[2]
    assert "Feature 2 (f2)" in out[3]
    assert "Feature 3 (f3)" in out[4]
    assert out[-1].startswith("total: 3")


def test_important_descriptors_sorting_and_output(monkeypatch):
    df = make_multi_feature_df()
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    model = ExtraTreesRegressor(n_estimators=5, random_state=0)
    result = important_descriptors(numeric_df=df, target="target", model=model)

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0][1] == "f1"
    assert result[1][1] == "f2"
    assert 0.0 <= result[0][0] <= 1.0
    assert 0.0 <= result[1][0] <= 1.0


def test_optimum_importance_returns_correct_labels(monkeypatch):
    df = make_multi_feature_df()
    sorted_list = [(1.0, "f1"), (0.1, "f2")]
    model = LinearRegression()
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    best = optimum_importance(
        numeric_df=df,
        all_descriptors=sorted_list,
        model=model,
        target="target",
    )
    assert best == ["f1", "f2"]


def test_print_loss_outputs_rmse(capsys):
    actual = np.array([0.0, 2.0, 4.0])
    pred = np.array([0.0, 1.0, 5.0])
    expected_rmse = np.sqrt(((0 - 0) ** 2 + (2 - 1) ** 2 + (4 - 5) ** 2) / 3)

    print_loss(actual, pred, unit="units")
    out = capsys.readouterr().out.strip()
    assert f"{expected_rmse:.3f}" in out
    assert out.endswith("units")


def test_single_descriptor_regression_and_plots(monkeypatch, capsys):
    df = pd.DataFrame({"x": [0, 1, 2, 3], "target": [0, 2, 4, 6]})
    mapping = {"x": "Feature X", "target": "Target"}

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    model = LinearRegression()

    single_descriptor_regression(
        numeric_df=df,
        descriptor="x",
        column_meaning_map=mapping,
        unit="units",
        target="target",
        model=model,
    )
    out = capsys.readouterr().out
    assert "root mean square error" in out


def test_compare_and_parity_plot_do_not_raise(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    y_train = np.array([1.0, 2.0, 3.0])
    y_pred_train = np.array([1.1, 2.1, 2.9])
    y_test = np.array([4.0, 5.0, 6.0])
    y_pred_test = np.array([3.9, 5.2, 6.1])
    score_train = "$R^2 = 0.99$"
    score_test = "$R^2 = 0.95$"
    unit = "units"

    compare(
        prediction_on_training=y_pred_train,
        prediction_on_test=y_pred_test,
        y_train=y_train,
        y_test=y_test,
        score_train=score_train,
        score_test=score_test,
        unit=unit,
    )

    parity_plot(
        y_train=y_train,
        y_pred_train=y_pred_train,
        y_test=y_test,
        y_pred_test=y_pred_test,
        r2_train=0.99,
        mae_train=0.05,
        r2_test=0.95,
        mae_test=0.1,
        unit=unit,
        target="dummy",
    )
