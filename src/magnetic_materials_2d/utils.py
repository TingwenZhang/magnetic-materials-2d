"""A collection of utility functions."""

# import python modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split

# global variables
TEST_SIZE = 0.2
RANDOM_STATE = 42


def sorted_descriptors(
    numeric_df: pd.DataFrame,
    target: str,
    model: None,
) -> list[tuple[float, str]]:
    """Sort all descriptors in decreasing score order, for a given model and
    target.

    Parameters
    ----------
    numeric_df: design matrix of numeric descriptors
    target: column label of the target attribute
    model: a model with fit() defined

    Effect
    ------
    Return a list of sorted descriptors as (score, column label)
    """
    descriptors = []

    y = numeric_df[target]
    y = np.asarray(y)

    # obtain all R^2
    for col in numeric_df.columns:
        if col == target:
            continue
        else:
            X = numeric_df[[col]]
            X = np.asarray(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
            reg = model.fit(X_train, y_train)
            descriptors.append((reg.score(X_test, y_test), col))

    descriptors.sort(reverse=True)

    return descriptors


def top12(
    descriptors: np.ndarray,
    column_meaning_map: dict[str, str],
    target: str,
    method: str,
):
    """Print the 12 highest scored descriptors for a given target and
    method."""
    title = (
        f"12 highest scored descriptors for {target} using {method} regression"
    )
    print(title)
    print("-" * len(title))
    for i in range(12):
        score = descriptors[i][0]
        label = descriptors[i][1]
        print(f"{i+1:2.0f}. {column_meaning_map[label]} (score = {score:.3f})")


def top_descriptors(
    descriptors: list[tuple[float, str]], threshold: float
) -> list[str]:
    """Select descriptors above the given threshold.

    Parameters
    ----------
    descriptors: a sorted list of descriptors
    threshold: threshold score

    Effects
    -------
    Return all descriptors above the threshold score
    """
    output = []
    for descriptor in descriptors:
        score = descriptor[0]
        if score > threshold:
            label = descriptor[1]
            output.append(label)
        else:
            break
    return output


def best_descriptors(
    numeric_df: pd.DataFrame,
    all_descriptors: list[tuple[float, str]],
    model: None,
    target: str,
) -> list[str]:
    """Select the best set descriptors for a given method, by searching through
    a range of threshold scores.

    Parameters
    ----------
    numeric_df: design matrix of numeric descriptors
    all_descriptors: a list of sorted descriptors
    model: a model with fit() defined
    target: column label of the target attribute

    Effects
    -------
    Return the list of best descriptors for the model and target
    """
    y = numeric_df[target]
    y = np.asarray(y)

    # try every 0.1 step size score between 0.0 and 1.0
    thresholds = np.linspace(0.0, 1.0, 11)
    max_score = 0.0
    solution = []
    scores = []
    optimum_threshold = 0.0
    for threshold in thresholds:
        descriptors = top_descriptors(all_descriptors, threshold)
        X = numeric_df[descriptors]
        X = np.asarray(X)
        if X.shape[1] == 0:  # no descriptors are above this threshold
            scores.append(0.0)
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        regressor = model.fit(X_train, y_train)
        score = regressor.score(X_test, y_test)
        scores.append(score)
        if score > max_score:
            max_score = score
            solution = descriptors
            optimum_threshold = threshold

    # visualize the variation in scores with thresholds
    plt.scatter(thresholds, scores)
    plt.xlabel("Threshold score")
    plt.ylabel("Score")
    plt.grid()
    plt.show()

    print("Optimum threshold = {:.3f}".format(optimum_threshold))
    print("Score = {:.3f}".format(max_score))
    print()
    return solution


def print_best_descriptors(
    descriptors: list[str],
    column_meaning_map: dict[str, str],
    target: str,
    method: str,
):
    """Print the best descriptors of a prediction model and target.

    Parameters
    ----------
    descriptors: best descriptors
    column_meaning_map: a dictionary mapping column labels to their description
    target: column label of target attribute
    method: name of method

    Effects
    -------
    Format the top descriptors nicely.
    """
    title = f"Best descriptors for {column_meaning_map[target]} using {method}"

    print(title)
    print("-" * len(title))
    for descriptor in descriptors:
        print(f"{column_meaning_map[descriptor]} ({descriptor})")
    print("total:", len(descriptors))


def important_descriptors(
    numeric_df: pd.DataFrame, target: str, model: None
) -> list[tuple[float, str]]:
    """Show the important descriptors selected by the extra trees regressor.

    Parameters
    ----------
    numeric_df: design matrix of numeric descriptors
    model: a model with feature_importances_
    target: column label of the target attribute

    Effects
    -------
    Return a sorted list of important descriptors for given method and target
    """
    X = numeric_df.drop(columns=[target])
    descriptors = X.columns
    # print(len(descriptors))
    X = np.asarray(X)
    y = numeric_df[target]
    y = np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=RANDOM_STATE
    )
    regressor = model.fit(X_train, y_train)
    feature_importances = regressor.feature_importances_
    # print(feature_importances)
    # print(len(feature_importances))

    n = len(numeric_df.columns) - 1
    list = []
    for i in range(n):
        list.append((feature_importances[i], descriptors[i]))
    list.sort(reverse=True)

    for tuple in list:
        descriptor = tuple[1]
        importance = tuple[0]
        print(f"({descriptor}, {importance:.3f})")

    return list


def optimum_importance(
    numeric_df: pd.DataFrame,
    all_descriptors: list[tuple[float, str]],
    model: None,
    target: str,
) -> list[str]:
    """Select the best descriptors for extra trees regressor, by searching
    through a range of importances.

    Parameters
    ----------
    numeric_df: design matrix of numeric descriptors
    all_descriptors: a list of sorted descriptors
    model: a model with fit() defined
    target: column label of the target attribute

    Effects
    -------
    Return the list of most important descriptors for the target.
    """
    y = numeric_df[target]
    y = np.asarray(y)

    # try every 0.01 step size importance between 0.0 and 0.5
    importances = np.linspace(0.0, 0.5, 51)
    max_score = 0.0
    solution = []
    scores = []
    optimum_importance = 0.0
    for importance in importances:
        descriptors = top_descriptors(all_descriptors, importance)
        X = numeric_df[descriptors]
        X = np.asarray(X)
        if X.shape[1] == 0:  # no descriptors are above this importance
            scores.append(0.0)
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        regressor = model.fit(X_train, y_train)
        score = regressor.score(X_test, y_test)
        scores.append(score)
        if score > max_score:
            max_score = score
            solution = descriptors
            optimum_importance = importance

    # visualize the variation in scores with importances
    plt.scatter(importances, scores)
    plt.xlabel("Threshold importance")
    plt.ylabel("Score")
    plt.grid()
    plt.show()

    print("Optimum importance = {:.3f}".format(optimum_importance))
    print("Score = {:.3f}".format(max_score))
    print()
    return solution


def print_loss(actual_y: np.ndarray, predicted_y: np.ndarray, unit: str):
    """Print root mean square error.

    Parameters
    ----------
    actual_y: ground-truth target values
    predicted_y: predicted target values
    unit: unit of target

    Effects
    -------
    Print root mean square error.
    """

    mse = np.mean((actual_y - predicted_y) ** 2)
    rmse = np.sqrt(mse)

    print(f"root mean square error = {rmse:.3f} {unit}")


def single_descriptor_regression(
    numeric_df: pd.DataFrame,
    descriptor: str,
    column_meaning_map: dict[str:str],
    unit: str,
    target: str,
    model: None,
) -> None:
    """Print the regression information using a descriptor and a model.

    Parameters
    ----------
    numeric_df: design matrix of numeric attributes
    descrpitor: column label of the descriptor in the dataframe
    column_meaning_map: a dictionary mapping column labels to their description
    unit: unit of target
    target: column label of target attribute
    model: a model with fit() defined
    """
    X = numeric_df[[descriptor]]
    X = np.asarray(X)
    y = numeric_df[target]
    y = np.asarray(y)
    regressor = model.fit(X, y)
    R_squared = f"$R^2 = {regressor.score(X, y):.3f}$"

    predicted_y = regressor.predict(X)

    print_loss(y, predicted_y, unit)

    # Plot your model alongside the X and y data.
    # type code here...
    plt.scatter(X, y, c="tab:red")
    plt.plot(X, predicted_y, c="tab:blue")
    plt.xlabel(column_meaning_map[descriptor])
    plt.ylabel(column_meaning_map[target])
    plt.text(
        0.05,
        0.8,
        R_squared,
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="blue", alpha=0.3),
    )
    plt.grid()
    plt.show()


def compare(
    prediction_on_training: np.ndarray,
    prediction_on_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    score_train: str,
    score_test: str,
    unit: str,
) -> None:
    """A helper function to visualize predication and actual values of the
    target attribute. A 45-degree line is plotted to show perfect prediction.

    Parameters
    ----------
    prediction_on_training: 1D NumPy array of predictions on the target attribute
    prediction_on_test: 1D NumPy array of predictions on the target attribute
    y_train: 1D NumPy array of actual values of the target attribute for the
    training set
    y_test: 1D NumPy array of actual values of the target attribute for the
    test set
    score_train: score on the training data formatted as a string
    score_test: score on the test data formatted as a string
    unit: unit of target
    """
    plt.figure(figsize=(6, 12))
    plt.subplot(2, 1, 1)
    plt.scatter(y_train, prediction_on_training, c="tab:blue")
    plt.plot(
        [y_train.min(), y_train.max()],
        [y_train.min(), y_train.max()],
        c="tab:red",
        linestyle="--",
    )
    plt.ylabel("predicted " + unit)
    plt.xlabel("actual " + unit)
    plt.title("Prediction on training data")
    plt.text(
        0.05,
        0.8,
        score_train,
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="blue", alpha=0.3),
    )
    plt.grid()
    plt.show()

    plt.figure(figsize=(6, 12))
    plt.subplot(2, 1, 2)
    plt.scatter(y_test, prediction_on_test, c="tab:blue")
    plt.plot(
        [y_train.min(), y_train.max()],
        [y_train.min(), y_train.max()],
        c="tab:red",
        linestyle="--",
    )
    plt.xlabel("actual " + unit)
    plt.ylabel("predicted " + unit)
    plt.title("Prediction on test data")
    plt.text(
        0.05,
        0.8,
        score_test,
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="blue", alpha=0.3),
    )
    plt.grid()
    plt.show()


def parity_plot(
    y_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    r2_train: float,
    mae_train: float,
    r2_test: float,
    mae_test: float,
    unit: str,
    target: str,
):
    """Plot both train and test predictions vs. actuals on one parity plot.

    Parameters
    ----------
    y_train : np.ndarray
        True target values for the training set.
    y_pred_train : np.ndarray
        Predicted target values for the training set.
    y_test : np.ndarray
        True target values for the test set.
    y_pred_test : np.ndarray
        Predicted target values for the test set.
    r2_train : float
        R² score on the training set.
    mae_train : float
        Mean absolute error on the training set.
    r2_test : float
        R² score on the test set.
    mae_test : float
        Mean absolute error on the test set.
    unit : str
        Unit of the target variable (for axis labels and legend).
    target : str
        Name of the target variable (for plot title).
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # major ticks
    ax.tick_params(
        axis="both", which="major", labelsize=14, width=1.5, length=6
    )
    # minor ticks
    ax.minorticks_on()
    ax.tick_params(axis="both", which="minor", labelsize=0, width=1, length=3)

    # 45° reference
    lo = min(y_train.min(), y_test.min())
    hi = max(y_train.max(), y_test.max())
    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=2,
        color="black",
        zorder=0,
    )

    # scatter
    ax.scatter(
        y_train,
        y_pred_train,
        marker="o",
        c="green",
        alpha=0.7,
        s=75,
        label=f"Train (R²={r2_train:.3f}, MAE={mae_train:.3f} {unit})",
        zorder=2,
    )
    ax.scatter(
        y_test,
        y_pred_test,
        marker="s",
        c="red",
        alpha=0.7,
        s=75,
        label=f"Test  (R²={r2_test:.3f}, MAE={mae_test:.3f} {unit})",
        zorder=2,
    )

    # labels & title
    ax.set_xlabel(f"Actual {unit}", fontweight="bold", fontsize=14)
    ax.set_ylabel(f"Predicted {unit}", fontweight="bold", fontsize=14)
    ax.set_title(f"Parity Plot for {target}", fontweight="bold", fontsize=16)

    # legend
    leg = ax.legend(frameon=True, fontsize=12)
    leg.get_frame().set_linewidth(1.5)
    leg.get_frame().set_edgecolor("black")

    # grid: major & minor
    ax.grid(which="major", linestyle="-", linewidth=1, alpha=0.6)
    ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.4)

    # equal aspect
    ax.set_aspect("equal", "box")

    # thick border
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color("black")

    plt.tight_layout()
    plt.show()


def test_performance(
    numeric_df: pd.DataFrame,
    descriptors: list[str],
    unit: str,
    target: str,
    model,
):
    """Train `model` on `descriptors`, report R² & MAE, and plot parity.

    Parameters
    ----------
    numeric_df : pd.DataFrame
        DataFrame containing both feature columns and the target column.
    descriptors : list[str]
        List of column names in `numeric_df` to use as input features.
    unit : str
        Unit of the target variable (for printed metrics and axis labels).
    target : str
        Column name of the target variable in `numeric_df`.
    model : estimator
        An unfitted scikit-learn–style regressor with `.fit()` and `.predict()`.
    """
    # 1) assemble data
    X = numeric_df[descriptors].to_numpy()
    y = numeric_df[target].to_numpy()

    # 2) split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 3) fit & predict
    reg = model.fit(X_train, y_train)
    y_pred_train = reg.predict(X_train)
    y_pred_test = reg.predict(X_test)

    # 4) compute metrics
    from sklearn.metrics import r2_score, mean_absolute_error

    r2_tr = r2_score(y_train, y_pred_train)
    mae_tr = mean_absolute_error(y_train, y_pred_train)
    r2_te = r2_score(y_test, y_pred_test)
    mae_te = mean_absolute_error(y_test, y_pred_test)

    # 5) print summary
    print(f"Training   →  R^2 = {r2_tr:.3f},  MAE = {mae_tr:.3f} {unit}")
    print(f"Test       →  R^2 = {r2_te:.3f},  MAE = {mae_te:.3f} {unit}")

    # 6) combined parity plot
    parity_plot(
        y_train,
        y_pred_train,
        y_test,
        y_pred_test,
        r2_tr,
        mae_tr,
        r2_te,
        mae_te,
        unit,
        target,
    )
