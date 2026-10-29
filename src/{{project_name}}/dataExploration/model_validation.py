"""Compare prediction distributions; use held-out data for model evaluation."""

from collections.abc import Sequence
from typing import Protocol

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .exploration import draw_central_tendency_measures


class Predictor(Protocol):
    """The prediction interface used by scikit-learn and statsmodels models."""

    def predict(self, data: pd.DataFrame):
        """Return one prediction per input row."""
        ...


def models_results(
    df: pd.DataFrame,
    list_of_models: Sequence[Predictor],
    actual_value_col: str,
    model_cols: Sequence[str] | None = None,
    *,
    feature_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return actual values and named predictions, retaining the input index.

    By default the target is removed before prediction. Supply feature_cols
    when models expect a particular set or order of input features.
    """
    if not df.columns.is_unique:
        raise ValueError("DataFrame columns must be unique")
    if actual_value_col not in df.columns:
        raise ValueError(f"Unknown actual_value_col: {actual_value_col}")
    names = (
        list(model_cols)
        if model_cols is not None
        else [f"model_{i}" for i in range(1, len(list_of_models) + 1)]
    )
    if len(names) != len(list_of_models):
        raise ValueError("model_cols must contain one name per model")
    if len(set(names)) != len(names) or actual_value_col in names:
        raise ValueError("Model names must be unique and different from the target")
    if feature_cols is not None and actual_value_col in feature_cols:
        raise ValueError("feature_cols must not include the target column")
    features = (
        df.loc[:, list(feature_cols)]
        if feature_cols is not None
        else df.drop(columns=actual_value_col)
    )
    result = df[[actual_value_col]].copy()
    for name, model in zip(names, list_of_models, strict=True):
        predictions = model.predict(features)
        # Assign positionally: a predictor's Series may have a different index.
        if isinstance(predictions, pd.Series):
            predictions = predictions.to_numpy()
        result[name] = predictions
    return result


def plot_model_predictions(
    models_results: pd.DataFrame,
    actual_value_col: str = "actual_value",
    model_cols: Sequence[str] | None = None,
    *,
    bins: int = 10,
    col_wrap: int = 2,
) -> sns.FacetGrid:
    """Return histograms comparing actual values with each model's predictions.

    Pass actual_value_col when the target has another name. All other columns
    are treated as predictions unless model_cols is supplied. Call plt.show()
    in the notebook, or save the returned grid with grid.savefig(...).
    """
    if actual_value_col not in models_results:
        raise ValueError(f"Unknown actual_value_col: {actual_value_col}")
    names = (
        list(model_cols)
        if model_cols is not None
        else [col for col in models_results if col != actual_value_col]
    )
    if not names or actual_value_col in names or len(set(names)) != len(names):
        raise ValueError("Select unique prediction columns, excluding the target")
    missing = set(names).difference(models_results.columns)
    if missing:
        raise ValueError(f"Unknown model_cols: {sorted(missing)}")
    # Separate plotting metadata from arbitrary target and prediction names.
    long_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "actual": models_results[actual_value_col].to_numpy(),
                    "predicted": models_results[name].to_numpy(),
                    "model": name,
                }
            )
            for name in names
        ],
        ignore_index=True,
    )
    grid = sns.FacetGrid(
        long_frame,
        col="model",
        col_wrap=col_wrap,
        sharex=False,
        sharey=False,
        height=4,
        aspect=1.5,
    )

    def draw_distributions(data: pd.DataFrame, **kwargs) -> None:
        ax = plt.gca()
        sns.histplot(
            data=data,
            x="actual",
            bins=bins,
            ax=ax,
            color="tab:blue",
            alpha=0.4,
            label="Actual",
        )
        sns.histplot(
            data=data,
            x="predicted",
            bins=bins,
            ax=ax,
            color="tab:orange",
            alpha=0.4,
            label="Predicted",
        )
        draw_central_tendency_measures(data["predicted"], ax, include_mode=False)
        ax.legend()

    grid.map_dataframe(draw_distributions)
    grid.set_titles("{col_name}")
    grid.tight_layout()
    return grid
