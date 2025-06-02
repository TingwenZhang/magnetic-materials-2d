# tests/test_hyper_search.py

import pytest
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

# Ensure `src/` is on PYTHONPATH so we can import from magnetic_materials_2d
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from magnetic_materials_2d.hyper_search import hyper_search_2d, best_hyperparameters


def make_tiny_dataframe():
    """Create a tiny DataFrame where 'feat' ranges 0–3 and 'target' equals 'feat'."""
    return pd.DataFrame({
        "feat":   [0, 1, 2, 3],
        "target": [0, 1, 2, 3]
    })


def test_hyper_search_2d_returns_correct_shapes_and_types():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 1, 2, 3])

    X_train, y_train = X[:2], y[:2]
    X_val,   y_val   = X[2:], y[2:]

    param_grid = {
        "max_depth":    [1],
        "n_estimators": [2]
    }

    vals1, vals2, train_scores, val_scores = hyper_search_2d(
        X_train, y_train, X_val, y_val,
        param_grid=param_grid,
        model_cls=RandomForestRegressor,
        fixed_params={}
    )

    assert isinstance(vals1, np.ndarray)
    assert isinstance(vals2, np.ndarray)
    assert isinstance(train_scores, np.ndarray)
    assert isinstance(val_scores, np.ndarray)

    assert vals1.shape == (1,)
    assert vals2.shape == (1,)
    assert train_scores.shape == (1, 1)
    assert val_scores.shape == (1, 1)

    assert np.all(np.isfinite(train_scores))
    assert np.all(np.isfinite(val_scores))
    assert train_scores.dtype.kind == "f"
    assert val_scores.dtype.kind == "f"


def test_hyper_search_2d_raises_value_error_for_wrong_param_grid():
    X = np.array([[0], [1]])
    y = np.array([0, 1])

    bad_grid_too_many = {
        "max_depth": [1, 2],
        "n_estimators": [1, 2],
        "min_samples_split": [2, 5]
    }
    with pytest.raises(ValueError):
        hyper_search_2d(X, y, X, y, param_grid=bad_grid_too_many, model_cls=RandomForestRegressor)

    bad_grid_too_few = {"max_depth": [1, 2]}
    with pytest.raises(ValueError):
        hyper_search_2d(X, y, X, y, param_grid=bad_grid_too_few, model_cls=RandomForestRegressor)


def test_best_hyperparameters_single_model_within_custom_ranges():
    df = make_tiny_dataframe()
    custom = {
        "max_depth":    [1, 2],
        "n_estimators": [1, 5]
    }

    best_depth, best_trees = best_hyperparameters(
        df=df,
        descriptors=["feat"],
        target="target",
        model_classes=[RandomForestRegressor],
        tune_params=("max_depth", "n_estimators"),
        custom_ranges=custom,
        plot_heatmap=False,
        plot_train=False,
        plot_val=False
    )

    assert best_depth in custom["max_depth"]
    assert best_trees in custom["n_estimators"]


def test_best_hyperparameters_multi_model_returns_dict_and_values_in_range():
    df = make_tiny_dataframe()
    custom = {
        "max_depth":    [1],
        "n_estimators": [2, 3]
    }

    result = best_hyperparameters(
        df=df,
        descriptors=["feat"],
        target="target",
        model_classes=[RandomForestRegressor, ExtraTreesRegressor],
        tune_params=("max_depth", "n_estimators"),
        custom_ranges=custom,
        plot_heatmap=False,
        plot_train=False,
        plot_val=False
    )

    assert isinstance(result, dict)
    assert "RandomForestRegressor" in result
    assert "ExtraTreesRegressor" in result

    rf_best = result["RandomForestRegressor"]
    et_best = result["ExtraTreesRegressor"]
    assert isinstance(rf_best, tuple) and len(rf_best) == 2
    assert isinstance(et_best, tuple) and len(et_best) == 2

    assert rf_best[0] in custom["max_depth"] and rf_best[1] in custom["n_estimators"]
    assert et_best[0] in custom["max_depth"] and et_best[1] in custom["n_estimators"]


def test_best_hyperparameters_missing_custom_range_raises_key_error():
    df = make_tiny_dataframe()
    with pytest.raises(KeyError):
        best_hyperparameters(
            df=df,
            descriptors=["feat"],
            target="target",
            model_classes=[RandomForestRegressor],
            tune_params=("nonexistent_param", "n_estimators"),
            custom_ranges=None,
            plot_heatmap=False,
            plot_train=False,
            plot_val=False
        )
