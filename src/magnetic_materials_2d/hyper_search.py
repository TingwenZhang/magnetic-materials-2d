"""A module for hyper-parameter searches."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from typing import Type, Iterable, Dict, Tuple, Union

# global variables
TEST_SIZE = 0.2
RANDOM_STATE = 42


def hyper_search_2d(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_grid: Dict[str, Iterable],
    model_cls: Type,
    fixed_params: Dict = {},
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Grid-search *exactly two* hyperparameters via brute-force.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix of shape (n_samples_train, n_features).
    y_train : np.ndarray
        Training target array of shape (n_samples_train,).
    X_val : np.ndarray
        Validation feature matrix of shape (n_samples_val, n_features).
    y_val : np.ndarray
        Validation target array of shape (n_samples_val,).
    param_grid : Dict[str, Iterable]
        Dictionary with exactly two keys.
        Each key is a hyperparameter name, and its value is an iterable
        of possible settings to try.
    model_cls : Type
        A scikit-learn–style estimator class (e.g. `RandomForestRegressor`).
        Will be instantiated via `model_cls(**cfg)`.
    fixed_params : Dict, optional
        Any parameters to pass *unchanged* to every model instantiation.

    Returns
    -------
    vals1 : np.ndarray
        1D array of the first hyperparameter’s tried values.
    vals2 : np.ndarray
        1D array of the second hyperparameter’s tried values.
    train_scores : np.ndarray
        2D array of shape (len(vals1), len(vals2)) containing R² scores
        on the training set.
    val_scores : np.ndarray
        2D array of shape (len(vals1), len(vals2)) containing R² scores
        on the validation set.

    Raises
    ------
    ValueError
        If `param_grid` does not contain exactly two entries.
    """
    if len(param_grid) != 2:
        raise ValueError(
            "param_grid must contain exactly two parameters to tune."
        )

    # pull out exactly two (param_name, values) pairs
    items = list(param_grid.items())
    (p1, vals1), (p2, vals2) = items[0], items[1]
    vals1 = np.array(vals1)
    vals2 = np.array(vals2)

    print(f"Evaluate the following values for {p1}: {vals1.tolist()}")
    print(f"Evaluate the following values for {p2}: {vals2.tolist()}")

    m, n = len(vals1), len(vals2)
    train_scores = np.zeros((m, n))
    val_scores = np.zeros((m, n))

    for i, v1 in enumerate(vals1):
        for j, v2 in enumerate(vals2):
            cfg = {
                **fixed_params,
                p1: v1,
                p2: v2,
                "random_state": RANDOM_STATE,
            }
            model = model_cls(**cfg)
            model.fit(X_train, y_train)
            train_scores[i, j] = model.score(X_train, y_train)
            val_scores[i, j] = model.score(X_val, y_val)

    return vals1, vals2, train_scores, val_scores


def best_hyperparameters(
    df: pd.DataFrame,
    descriptors: list[str],
    target: str,
    model_classes: list[Type] = [RandomForestRegressor, ExtraTreesRegressor],
    tune_params: Tuple[str, str] = ("max_depth", "n_estimators"),
    custom_ranges: Dict[str, Iterable] = None,
    plot_heatmap: bool = True,
    plot_train: bool = True,
    plot_val: bool = True,
) -> Union[Dict[str, Tuple], Tuple]:
    """Tune two hyperparameters for one or more models and return the best
    settings.

    For each estimator in `model_classes`, performs a 2-D grid search over
    the two hyperparameters named in `tune_params`.  Optionally shows heatmaps
    of training/validation R².

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature columns and the target column.
    descriptors : list[str]
        List of column names in `df` to use as features.
    target : str
        Name of the column in `df` to use as the regression target.
    model_classes : list[Type], optional
        List of estimator classes to tune.  Default: `[RandomForestRegressor, ExtraTreesRegressor]`.
    tune_params : Tuple[str, str], optional
        Pair of hyperparameter names to grid-search.  Default: `("max_depth", "n_estimators")`.
    custom_ranges : Dict[str, Iterable], optional
        If provided, must map each name in `tune_params` to an iterable of values.
        Overrides the built-in defaults.
    plot_heatmap : bool, optional
        If True, draw heatmaps for both train and validation scores.
    plot_train : bool, optional
        If True, plot the training R² heatmap.
    plot_val : bool, optional
        If True, plot the validation R² heatmap.

    Returns
    -------
    Dict[str, Tuple]
        If `len(model_classes) > 1`, returns a dict mapping each model’s
        `.__name__` to a `(best_p1, best_p2)` tuple.
    Tuple
        If exactly one class was passed in `model_classes`, returns its
        best-param tuple directly.

    Raises
    ------
    KeyError
        If one of the names in `tune_params` is not found in the default
        ranges and not supplied in `custom_ranges`.
    """
    # default hyperparameter ranges
    defaults = {
        "max_depth": np.arange(1, 26),
        "n_estimators": np.arange(25, 301, 25),
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "min_weight_fraction_leaf": [0.0, 0.01, 0.05],
        "max_features": ["sqrt", "log2", None]
        + [i / 10 for i in range(1, 11)],
        "max_leaf_nodes": [None, 10, 20, 50],
        "min_impurity_decrease": [0.0, 0.01, 0.05],
        "ccp_alpha": [0.0, 0.001, 0.01],
    }

    p1, p2 = tune_params
    grid: Dict[str, Iterable] = {}
    for p in (p1, p2):
        if custom_ranges and p in custom_ranges:
            grid[p] = custom_ranges[p]
        elif p in defaults:
            grid[p] = defaults[p]
        else:
            raise KeyError(
                f"No default range for '{p}'. Provide it in custom_ranges."
            )

    # split data
    X = df[descriptors].to_numpy()
    y = df[target].to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results: Dict[str, Tuple] = {}
    for cls in model_classes:
        v1s, v2s, tr_scores, val_scores = hyper_search_2d(
            X_train, y_train, X_val, y_val, param_grid=grid, model_cls=cls
        )

        # best on validation
        idx_flat = np.argmax(val_scores)
        i_best, j_best = np.unravel_index(idx_flat, val_scores.shape)
        best_p1, best_p2 = v1s[i_best], v2s[j_best]
        best_score = val_scores[i_best, j_best]

        print(
            f"{cls.__name__:>20}  best R²={best_score:.3f}  "
            f"@ {p1}={best_p1}, {p2}={best_p2}"
        )

        # optional heatmaps
        if plot_heatmap:
            if plot_train:
                plt.figure()
                plt.title(f"{cls.__name__} training R²")
                plt.xlabel(p2)
                plt.ylabel(p1)
                plt.imshow(
                    tr_scores,
                    aspect="auto",
                    origin="lower",
                    extent=[v2s.min(), v2s.max(), v1s.min(), v1s.max()],
                )
                plt.colorbar(label=r"$R^2$ (train)")
                plt.show()
            if plot_val:
                plt.figure()
                plt.title(f"{cls.__name__} validation R²")
                plt.xlabel(p2)
                plt.ylabel(p1)
                plt.imshow(
                    val_scores,
                    aspect="auto",
                    origin="lower",
                    extent=[v2s.min(), v2s.max(), v1s.min(), v1s.max()],
                )
                plt.colorbar(label=r"$R^2$ (val)")
                plt.show()

        results[cls.__name__] = (best_p1, best_p2)

    # unwrap single-model case
    if len(model_classes) == 1:
        return next(iter(results.values()))
    return results
