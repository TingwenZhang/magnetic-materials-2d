"""A module for hyper-parameter searches."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from typing import Type, Iterable, Dict, Tuple

# global variables
TEST_SIZE = 0.2
RANDOM_STATE = 42


def hyper_search_2d(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    param_grid: Dict[str, Iterable],
    model_cls:  Type,
    fixed_params: Dict = {}
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Grid-search *exactly two* hyperparameters via brute-force.
    (unchanged)
    """
    if len(param_grid) != 2:
        raise ValueError("param_grid must contain exactly two parameters to tune.")

    (p1, p2), (vals1, vals2) = list(param_grid.items()), list(param_grid.values())
    vals1 = np.array(vals1)
    vals2 = np.array(vals2)

    m, n = len(vals1), len(vals2)
    train_scores = np.zeros((m, n))
    val_scores   = np.zeros((m, n))

    for i, v1 in enumerate(vals1):
        for j, v2 in enumerate(vals2):
            cfg = {**fixed_params, p1: v1, p2: v2, "random_state": RANDOM_STATE}
            model = model_cls(**cfg)
            model.fit(X_train, y_train)
            train_scores[i, j] = model.score(X_train, y_train)
            val_scores[i, j]   = model.score(X_val,   y_val)

    return vals1, vals2, train_scores, val_scores


def best_hyperparameters(
    df: pd.DataFrame,
    descriptors: list[str],
    target: str,
    model_classes: list[Type]           = [RandomForestRegressor, ExtraTreesRegressor],
    tune_params: Tuple[str, str]        = ("max_depth", "n_estimators"),
    custom_ranges: Dict[str, Iterable]  = None,
    plot_heatmap: bool                  = True,
    plot_train:    bool                  = True,
    plot_val:      bool                  = True
) -> Dict[str, Tuple]:
    """
    For each model in `model_classes`, tune the two hyperparameters in `tune_params`
    and return each model’s best-setting tuple.

    If plot_heatmap is True, draws heatmaps for training and/or validation.
    """
    # default ranges …
    defaults = {
        "max_depth":             np.arange(1, 26),
        "n_estimators":          np.arange(25, 301, 25),
        "min_samples_split":     [2, 5, 10, 20],
        "min_samples_leaf":      [1, 2, 4, 8],
        "min_weight_fraction_leaf": [0.0, 0.01, 0.05],
        "max_features":          ["sqrt", "log2", None] + list(np.linspace(0.1, 1.0, 10)), "max_leaf_nodes": [None, 10, 20, 50],
        "min_impurity_decrease": [0.0, 0.01, 0.05],
        "ccp_alpha":             [0.0, 0.001, 0.01],
    }

    p1, p2 = tune_params
    grid = {}
    for p in (p1, p2):
        if custom_ranges and p in custom_ranges:
            grid[p] = custom_ranges[p]
        elif p in defaults:
            grid[p] = defaults[p]
        else:
            raise KeyError(f"No default range for '{p}'. Provide it in custom_ranges.")

    X = df[descriptors].to_numpy()
    y = df[target].to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results = {}
    for cls in model_classes:
        v1s, v2s, tr_scores, val_scores = hyper_search_2d(
            X_train, y_train, X_val, y_val, param_grid=grid, model_cls=cls
        )

        # find best on validation
        idx_flat = np.argmax(val_scores)
        i_best, j_best = np.unravel_index(idx_flat, val_scores.shape)
        best_p1, best_p2 = v1s[i_best], v2s[j_best]
        best_score = val_scores[i_best, j_best]

        print(f"{cls.__name__:>20}  best R²={best_score:.3f}  "
              f"@ {p1}={best_p1}, {p2}={best_p2}")

        if plot_heatmap:
            # training heatmap
            if plot_train:
                plt.figure()
                plt.title(f"{cls.__name__} training R²")
                plt.xlabel(p2)
                plt.ylabel(p1)
                plt.imshow(tr_scores, aspect='auto', origin='lower',
                           extent=[v2s.min(), v2s.max(), v1s.min(), v1s.max()])
                plt.colorbar(label="R² (train)")
                plt.show()

            # validation heatmap
            if plot_val:
                plt.figure()
                plt.title(f"{cls.__name__} validation R²")
                plt.xlabel(p2)
                plt.ylabel(p1)
                plt.imshow(val_scores, aspect='auto', origin='lower',
                           extent=[v2s.min(), v2s.max(), v1s.min(), v1s.max()])
                plt.colorbar(label="R² (val)")
                plt.show()

        results[cls.__name__] = (best_p1, best_p2)

    return results

#example usage
#best = best_hyperparameters(
#    df,
#    descriptors=['feat1','feat2','feat3'],
#    target='y',
#    model_classes=[RandomForestRegressor],
#    tune_params=('max_depth', 'n_estimators'),
#    plot_heatmap=True,
#    plot_train=True,
#    plot_val=False    # only plot training heatmap
#)
