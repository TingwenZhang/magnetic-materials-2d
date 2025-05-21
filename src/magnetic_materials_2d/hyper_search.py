"""A module for hyper-paramaters searches."""

# import python modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import all machine learning functions
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# global variables
TEST_SIZE = 0.2
RANDOM_STATE = 42


def hyper_search_2d(X_train, y_train, X_val, y_val, size_1, size_2):
    """Function that searches evaluates a list of hyperparameters."""
    max_depth_values = np.arange(size_1) + 1
    n_estimators_values = np.arange(1, size_2, 25)  # save some iterations
    print("Evaluate the following values for max_depth: ", max_depth_values)
    print(
        "Evaluate the following values for n_estimators: ", n_estimators_values
    )

    m = len(max_depth_values)
    n = len(n_estimators_values)

    training_hyper_scores = np.zeros((m, n))
    test_hyper_scores = np.zeros((m, n))

    for ith, max_depth in enumerate(max_depth_values):
        for jth, n_estimators in enumerate(n_estimators_values):
            rf_model_i = RandomForestRegressor(
                max_depth=max_depth,
                n_estimators=n_estimators,
                random_state=RANDOM_STATE,
            )
            rf_model_i.fit(X_train, y_train)
            score = rf_model_i.score(X_train, y_train)
            training_hyper_scores[ith][jth] = score
            score = rf_model_i.score(X_val, y_val)
            test_hyper_scores[ith][jth] = score

    return (
        max_depth_values,
        n_estimators_values,
        training_hyper_scores,
        test_hyper_scores,
    )


# ==============================================================================
# ~1 minutes to 5 minutes to finish
# ==============================================================================
def best_hyperparameters(
    numeric_df: pd.DataFrame, descriptors: list[str], target: str
) -> tuple[int, int]:
    """Determine the best (max_depth, n_estimators).

    Parameters
    ----------
    numeric_df: design matrix of numeric attributes
    descriptors: list of the best descriptors
    target: column label of target attribute

    Effects
    -------
    Return the best max_depth, n_estimator
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
        X_train, y_train, X_test, y_test, max_depth_bound, n_estimators_bound
    )

    plt.imshow(training_hyper_scores)
    plt.colorbar()
    plt.xlabel("n_estimators/25")
    plt.ylabel("max_depth")
    plt.show()

    plt.imshow(test_hyper_scores)
    plt.colorbar()
    plt.xlabel("n_estimators/25")
    plt.ylabel("max_depth")
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
