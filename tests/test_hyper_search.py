# tests/test_hyper_search.py

import os
import sys
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

# Ensure “src” is on PYTHONPATH so we can import from magnetic_materials_2d
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from magnetic_materials_2d.hyper_search import (
    hyper_search_2d,
    best_hyperparameters,
)


def make_tiny_dataframe():
    """Create a tiny DataFrame where 'feat' ranges 0–3 and 'target' equals
    'feat'."""
    return pd.DataFrame({"feat": [0, 1, 2, 3], "target": [0, 1, 2, 3]})


def test_hyper_search_2d_returns_correct_shapes_and_types():
    """
    hyper_search_2d should return four numpy arrays:
      - vals1 (shape (m,))
      - vals2 (shape (n,))
      - train_scores (shape (m, n))
      - val_scores (shape (m, n))
    And all entries in train_scores and val_scores should be finite floats.
    """
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 1, 2, 3])

    # Split: first two for training, last two for validation
    X_train, y_train = X[:2], y[:2]
    X_val, y_val = X[2:], y[2:]

    param_grid = {"max_depth": [1], "n_estimators": [2]}

    vals1, vals2, train_scores, val_scores = hyper_search_2d(
        X_train,
        y_train,
        X_val,
        y_val,
        param_grid=param_grid,
        model_cls=RandomForestRegressor,
        fixed_params={},
    )

    # Only one choice per hyperparameter ⇒ shapes should be (1,) and (1,1)
    assert isinstance(vals1, np.ndarray)
    assert isinstance(vals2, np.ndarray)
    assert isinstance(train_scores, np.ndarray)
    assert isinstance(val_scores, np.ndarray)

    assert vals1.shape == (1,)
    assert vals2.shape == (1,)
    assert train_scores.shape == (1, 1)
    assert val_scores.shape == (1, 1)

    # Entries in train_scores and val_scores must be finite floats
    assert np.all(np.isfinite(train_scores))
    assert np.all(np.isfinite(val_scores))
    assert train_scores.dtype.kind == "f"
    assert val_scores.dtype.kind == "f"


def test_hyper_search_2d_raises_value_error_for_wrong_param_grid():
    """hyper_search_2d must raise ValueError if param_grid has not exactly two
    keys."""
    X = np.array([[0], [1]])
    y = np.array([0, 1])

    bad_grid_too_many = {
        "max_depth": [1, 2],
        "n_estimators": [1, 2],
        "min_samples_split": [2, 5],
    }
    with pytest.raises(ValueError):
        hyper_search_2d(
            X,
            y,
            X,
            y,
            param_grid=bad_grid_too_many,
            model_cls=RandomForestRegressor,
        )

    bad_grid_too_few = {"max_depth": [1, 2]}
    with pytest.raises(ValueError):
        hyper_search_2d(
            X,
            y,
            X,
            y,
            param_grid=bad_grid_too_few,
            model_cls=RandomForestRegressor,
        )


def test_best_hyperparameters_single_model_within_custom_ranges():
    """When only one model class is passed, best_hyperparameters should return
    a tuple (best_p1, best_p2), and both values must come from the provided
    custom_ranges."""
    df = make_tiny_dataframe()

    # Provide a custom range for max_depth and n_estimators
    custom = {"max_depth": [1, 2], "n_estimators": [1, 5]}

    best_depth, best_trees = best_hyperparameters(
        df=df,
        descriptors=["feat"],
        target="target",
        model_classes=[RandomForestRegressor],
        tune_params=("max_depth", "n_estimators"),
        custom_ranges=custom,
        plot_heatmap=False,
        plot_train=False,
        plot_val=False,
    )

    # The returned values must be from the sets we supplied
    assert best_depth in custom["max_depth"]
    assert best_trees in custom["n_estimators"]


def test_best_hyperparameters_multi_model_returns_dict_and_values_in_range():
    """When passing two model classes, best_hyperparameters should return a
    dict mapping each model’s __name__ to a (best_p1, best_p2) tuple.

    Each entry must lie within our custom_ranges.
    """
    df = make_tiny_dataframe()

    custom = {"max_depth": [1], "n_estimators": [2, 3]}

    result = best_hyperparameters(
        df=df,
        descriptors=["feat"],
        target="target",
        model_classes=[RandomForestRegressor, ExtraTreesRegressor],
        tune_params=("max_depth", "n_estimators"),
        custom_ranges=custom,
        plot_heatmap=False,
        plot_train=False,
        plot_val=False,
    )

    # Expect a dict with exactly two keys
    assert isinstance(result, dict)
    assert set(result.keys()) == {
        "RandomForestRegressor",
        "ExtraTreesRegressor",
    }

    for model_name, (depth, trees) in result.items():
        assert depth in custom["max_depth"]
        assert trees in custom["n_estimators"]


def test_best_hyperparameters_missing_custom_range_raises_key_error():
    """If tune_params names are not in defaults and custom_ranges is None (or
    missing the key), best_hyperparameters should raise KeyError."""
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
            plot_val=False,
        )
