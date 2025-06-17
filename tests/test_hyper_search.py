# tests/test_hyper_search.py

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from magnetic_materials_2d.hyper_search import (
    hyper_search_2d,
    best_hyperparameters,
    TEST_SIZE,
    RANDOM_STATE
)

@pytest.fixture
def synthetic_data():
    """Create synthetic regression data for testing."""
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test

@pytest.fixture
def synthetic_dataframe():
    """Create synthetic pandas DataFrame for testing."""
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=RANDOM_STATE)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df['target'] = y
    return df

def test_hyper_search_2d_shape(synthetic_data):
    """Test that hyper_search_2d returns arrays of correct shape."""
    X_train, X_test, y_train, y_test = synthetic_data
    max_depth_bound = 5
    n_estimators_bound = 50
    
    for model_name in ["random forest", "extra trees"]:
        result = hyper_search_2d(
            X_train, y_train, X_test, y_test,
            max_depth_bound, n_estimators_bound, model_name
        )
        
        max_depth_values, n_estimators_values, train_scores, test_scores = result
        
        assert len(max_depth_values) == max_depth_bound
        assert len(n_estimators_values) == len(range(1, n_estimators_bound, 25))
        assert train_scores.shape == (max_depth_bound, len(n_estimators_values))
        assert test_scores.shape == (max_depth_bound, len(n_estimators_values))

def test_hyper_search_2d_scores(synthetic_data):
    """Test that scores are within expected range (R^2 score between 0 and 1)."""
    X_train, X_test, y_train, y_test = synthetic_data
    max_depth_bound = 5
    n_estimators_bound = 50
    
    for model_name in ["random forest", "extra trees"]:
        _, _, train_scores, test_scores = hyper_search_2d(
            X_train, y_train, X_test, y_test,
            max_depth_bound, n_estimators_bound, model_name
        )
        
        assert np.all(train_scores >= 0) and np.all(train_scores <= 1)
        assert np.all(test_scores >= -1) and np.all(test_scores <= 1)  # Can be negative for bad models

def test_best_hyperparameters(synthetic_dataframe):
    """Test that best_hyperparameters returns valid values."""
    df = synthetic_dataframe
    descriptors = [col for col in df.columns if col.startswith('feature_')]
    target = 'target'
    
    for model_name in ["random forest", "extra trees"]:
        max_depth, n_estimators = best_hyperparameters(
            df, descriptors, target, model_name
        )
        
        assert isinstance(max_depth, int)
        assert isinstance(n_estimators, int)
        assert max_depth > 0 and max_depth <= 25
        assert n_estimators > 0 and n_estimators < 300

def test_hyper_search_model_types(synthetic_data):
    """Test that invalid model names raise an assertion error."""
    X_train, X_test, y_train, y_test = synthetic_data
    max_depth_bound = 5
    n_estimators_bound = 50
    
    with pytest.raises(AssertionError):
        hyper_search_2d(
            X_train, y_train, X_test, y_test,
            max_depth_bound, n_estimators_bound, "invalid model"
        )

def test_hyper_search_output_types(synthetic_data):
    """Test that output types are correct."""
    X_train, X_test, y_train, y_test = synthetic_data
    max_depth_bound = 5
    n_estimators_bound = 50
    
    result = hyper_search_2d(
        X_train, y_train, X_test, y_test,
        max_depth_bound, n_estimators_bound, "random forest"
    )
    
    assert isinstance(result[0], np.ndarray)
    assert isinstance(result[1], np.ndarray)
    assert isinstance(result[2], np.ndarray)
    assert isinstance(result[3], np.ndarray)