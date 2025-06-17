# tests/test_utils.py

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.datasets import make_regression
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
    test_performance,
    TEST_SIZE,
    RANDOM_STATE
)

# Fixtures
@pytest.fixture
def synthetic_data():
    """Create synthetic regression data for testing."""
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=RANDOM_STATE)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df['target'] = y
    return df

@pytest.fixture
def column_meaning_map():
    """Create a simple column meaning map."""
    return {
        'feature_0': 'Feature 0 Description',
        'feature_1': 'Feature 1 Description',
        'feature_2': 'Feature 2 Description',
        'feature_3': 'Feature 3 Description',
        'feature_4': 'Feature 4 Description',
        'target': 'Target Description'
    }

@pytest.fixture
def sorted_descriptors_list(synthetic_data):
    """Create sorted descriptors for testing."""
    model = LinearRegression()
    return sorted_descriptors(synthetic_data, 'target', model)

def test_sorted_descriptors(synthetic_data, sorted_descriptors_list):
    """Test sorted_descriptors returns correct format and ordering."""
    # Check return type
    assert isinstance(sorted_descriptors_list, list)
    
    # Check element types
    for item in sorted_descriptors_list:
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], float)
        assert isinstance(item[1], str)
    
    # Check ordering (descending)
    scores = [item[0] for item in sorted_descriptors_list]
    assert scores == sorted(scores, reverse=True)
    
    # Check all features are included
    feature_columns = [col for col in synthetic_data.columns if col != 'target']
    returned_features = [item[1] for item in sorted_descriptors_list]
    assert set(feature_columns) == set(returned_features)

def test_top12(capsys, sorted_descriptors_list, column_meaning_map):
    """Test top12 prints correct output."""
    top12(sorted_descriptors_list, column_meaning_map, 'target', 'Linear Regression')
    captured = capsys.readouterr()
    
    # Check title
    assert "12 highest scored descriptors for target using Linear Regression" in captured.out
    
    # Check 12 items are printed
    lines = [line for line in captured.out.split('\n') if line.strip()]
    assert len(lines) >= 13  # Title + 12 items + separator
    
    # Check item format
    for i in range(1, 13):
        assert f"{i}." in captured.out

def test_top_descriptors(sorted_descriptors_list):
    """Test top_descriptors returns correct descriptors."""
    # Test threshold above max score
    result = top_descriptors(sorted_descriptors_list, 1.5)
    assert len(result) == 0
    
    # Test threshold below min score
    min_score = min(item[0] for item in sorted_descriptors_list)
    result = top_descriptors(sorted_descriptors_list, min_score - 0.1)
    assert len(result) == len(sorted_descriptors_list)
    
    # Test valid threshold
    threshold = sorted_descriptors_list[2][0] - 0.01
    result = top_descriptors(sorted_descriptors_list, threshold)
    assert len(result) >= 3
    for item in result:
        assert item in [d[1] for d in sorted_descriptors_list]

def test_best_descriptors(synthetic_data, sorted_descriptors_list):
    """Test best_descriptors returns valid descriptors."""
    model = LinearRegression()
    result = best_descriptors(synthetic_data, sorted_descriptors_list, model, 'target')
    
    # Check return type and content
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(item, str) for item in result)
    
    # Check all descriptors are valid columns
    valid_columns = set(synthetic_data.columns) - {'target'}
    assert all(col in valid_columns for col in result)

def test_print_best_descriptors(capsys, column_meaning_map, sorted_descriptors_list):
    """Test print_best_descriptors prints correct output."""
    descriptors = [item[1] for item in sorted_descriptors_list[:3]]
    print_best_descriptors(descriptors, column_meaning_map, 'target', 'Linear Regression')
    captured = capsys.readouterr()
    
    # Check title
    assert "Best descriptors for Target Description using Linear Regression" in captured.out
    
    # Check descriptors are printed
    for desc in descriptors:
        assert column_meaning_map[desc] in captured.out

def test_important_descriptors(synthetic_data):
    """Test important_descriptors returns correct format."""
    model = ExtraTreesRegressor(random_state=RANDOM_STATE)
    result = important_descriptors(synthetic_data, 'target', model)
    
    # Check return type
    assert isinstance(result, list)
    
    # Check element types
    for item in result:
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], float)
        assert isinstance(item[1], str)
    
    # Check ordering (descending)
    importances = [item[0] for item in result]
    assert importances == sorted(importances, reverse=True)

def test_optimum_importance(synthetic_data):
    """Test optimum_importance returns valid descriptors."""
    model = ExtraTreesRegressor(random_state=RANDOM_STATE)
    all_descriptors = important_descriptors(synthetic_data, 'target', model)
    result = optimum_importance(synthetic_data, all_descriptors, model, 'target')
    
    # Check return type and content
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(item, str) for item in result)
    
    # Check all descriptors are valid columns
    valid_columns = set(synthetic_data.columns) - {'target'}
    assert all(col in valid_columns for col in result)

def test_print_loss(capsys):
    """Test print_loss prints correct RMSE."""
    actual_y = np.array([1, 2, 3, 4, 5])
    predicted_y = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    print_loss(actual_y, predicted_y, "eV")
    captured = capsys.readouterr()
    
    # Calculate expected RMSE
    expected_rmse = np.sqrt(np.mean((actual_y - predicted_y) ** 2))
    assert f"root mean square error = {expected_rmse:.3f} eV" in captured.out

def test_single_descriptor_regression(synthetic_data, column_meaning_map):
    """Test single_descriptor_regression runs without errors."""
    # Mock plt.show to prevent displaying plots
    plt.show = lambda: None
    
    model = LinearRegression()
    single_descriptor_regression(
        synthetic_data,
        'feature_0',
        column_meaning_map,
        'units',
        'target',
        model
    )

def test_compare():
    """Test compare runs without errors."""
    # Mock plt.show to prevent displaying plots
    plt.show = lambda: None
    
    y_train = np.array([1, 2, 3])
    y_test = np.array([4, 5])
    prediction_on_training = np.array([1.1, 1.9, 3.1])
    prediction_on_test = np.array([4.1, 4.9])
    
    compare(
        prediction_on_training,
        prediction_on_test,
        y_train,
        y_test,
        "R2=0.95",
        "R2=0.90",
        "eV"
    )

def test_parity_plot():
    """Test parity_plot runs without errors."""
    # Mock plt.show to prevent displaying plots
    plt.show = lambda: None
    
    y_train = np.array([1, 2, 3])
    y_pred_train = np.array([1.1, 1.9, 3.1])
    y_test = np.array([4, 5])
    y_pred_test = np.array([4.1, 4.9])
    
    parity_plot(
        y_train,
        y_pred_train,
        y_test,
        y_pred_test,
        0.95,
        0.1,
        0.90,
        0.15,
        "eV",
        "Target"
    )

def test_test_performance(synthetic_data):
    """Test test_performance runs without errors."""
    # Mock plt.show to prevent displaying plots
    plt.show = lambda: None
    
    descriptors = ['feature_0', 'feature_1', 'feature_2']
    model = LinearRegression()
    
    test_performance(
        synthetic_data,
        descriptors,
        "units",
        "target",
        model
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
