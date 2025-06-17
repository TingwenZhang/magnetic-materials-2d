"""A module for hyper-parameter searches."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict

# global variables
TEST_SIZE = 0.2
RANDOM_STATE = 42


def hyper_search_2d(X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_test: np.ndarray,
                    y_test: np.ndarray,
                    max_depth_bound: int,
                    n_estimators_bound: int,
                    model_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Brute-force search through max_depth and n_estimators.

    Parameters
    ----------
    X_train: training input
    y_train: traing ground truth
    X_test: test input,
    y_test: test ground truth
    max_depth_size: upper bound of max_depth
    n_estimators_size: uppor bound of n_estimators
    model_name: name of the model with max_depth and n_estimators

    Returns
    -------
    max_depth_values: max_depth searched
    n_estimators_values: n_estimators searched
    training_hyper_scores: 2D grid of scores on training data
    test_hyper_scores: 2D grid of scores on test data
    """
    max_depth_values = np.arange(max_depth_bound) + 1
    n_estimators_values = np.arange(1, n_estimators_bound, 25)
    print("Evaluate the following values for max_depth: ", max_depth_values)
    print("Evaluate the following values for n_estimators: ", n_estimators_values)

    m = len(max_depth_values)
    n = len(n_estimators_values)

    training_hyper_scores = np.zeros((m, n))
    test_hyper_scores = np.zeros((m, n))

    if model_name == "random forest":
        for i, max_depth in enumerate(max_depth_values):
            for j, n_estimators in enumerate(n_estimators_values):
                model = RandomForestRegressor(
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    random_state=RANDOM_STATE,
                )
                model.fit(X_train, y_train)
                score = model.score(X_train, y_train)
                training_hyper_scores[i][j] = score
                score = model.score(X_test, y_test)
                test_hyper_scores[i][j] = score
    else:
        assert(model_name == "extra trees")
        for i, max_depth in enumerate(max_depth_values):
            for j, n_estimators in enumerate(n_estimators_values):
                model = ExtraTreesRegressor(
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    random_state=RANDOM_STATE,
                )
                model.fit(X_train, y_train)
                score = model.score(X_train, y_train)
                training_hyper_scores[i][j] = score
                score = model.score(X_test, y_test)
                test_hyper_scores[i][j] = score

    return (
        max_depth_values,
        n_estimators_values,
        training_hyper_scores,
        test_hyper_scores
    )


# ==============================================================================
# ~3 minutes to 5 minutes to finish
# ==============================================================================
def best_hyperparameters(
    numeric_df: pd.DataFrame, descriptors: list[str], target: str, model_name: str
) -> tuple[int, int]:
    """Determine the best (max_depth, n_estimators).

    Parameters
    ----------
    numeric_df: design matrix of numeric attributes
    descriptors: list of the best descriptors
    target: column label of target attribute
    model_name: name of the model with max_depth and n_estimators

    Returns
    -------
    best (max_depth, n_estimator)
    """
    max_depth_bound = 25
    n_estimators_bound = 300
    X = numeric_df[descriptors]
    X = np.asarray(X)
    y = numeric_df[target]
    y = np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    (
        max_depth_values,
        n_estimators_values,
        training_hyper_scores,
        test_hyper_scores,
    ) = hyper_search_2d(
        X_train, y_train, X_test, y_test, max_depth_bound, n_estimators_bound, model_name
    )

    plt.imshow(training_hyper_scores)
    plt.colorbar()
    plt.xlabel("n_estimators/25")
    plt.ylabel("max_depth")
    plt.title(f"{target} ({model_name} on training)")
    plt.show()

    plt.imshow(test_hyper_scores)
    plt.colorbar()
    plt.xlabel("n_estimators/25")
    plt.ylabel("max_depth")
    plt.title(f"{target} ({model_name} on test)")
    plt.show()

    max_depth = 0
    n_estimators = 0
    max_score = 0
    for i in range(len(max_depth_values)):
        for j in range(len(n_estimators_values)):
            if test_hyper_scores[i][j] > max_score:
                max_score = test_hyper_scores[i][j]
                max_depth = max_depth_values[i]
                n_estimators = n_estimators_values[j]

    print("max score =", max_score)

    print(
        "Best hyperparameter (max_depth, n_estimators): (",
        max_depth,
        ",",
        n_estimators,
        ")",
    )

    return (max_depth, n_estimators)
