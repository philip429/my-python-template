import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import upsetplot
from sklearn.impute import KNNImputer

from .exploration import decode_categories, encode_categories


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and a boolean flag for each original column."""
    counts = df.isna().sum()
    return pd.DataFrame(
        {
            "column_name": df.columns.to_list(),
            "missing_count": counts.to_numpy(),
            "has_null": counts.to_numpy() > 0,
        }
    )


def missing_summary_plot(df: pd.DataFrame) -> sns.FacetGrid:
    """Plot the proportions of observed and missing values per variable."""

    return (
        df.isnull()
        .melt(var_name="variable", value_name="value")
        .pipe(
            lambda df: sns.displot(
                data=df,
                y="variable",
                hue="value",
                multiple="fill",
                aspect=2,
            )
        )
    )


def missing_heatmap(df: pd.DataFrame):
    """Return a heatmap Axes showing the location of missing observations."""

    return (
        df.isnull()
        .transpose()
        .pipe(
            lambda df: sns.heatmap(
                data=df, cmap="viridis", cbar=False, yticklabels=True, xticklabels=False
            )
        )
    )


missing_heatMap = missing_heatmap  # Compatibility with existing notebooks.


def impute_knn(
    df: pd.DataFrame,
    n_neighbors: int = 5,
    method: str = "ordinal",
    *,
    exclude_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Impute a frame for exploration and return decoded category labels.

    encode_categories validates types and preserves dates/durations. KNN runs
    only on numeric output columns. Non-numeric columns, including NaT, pass
    through unchanged. Use exclude_columns for identifiers and targets. Input
    is unchanged. Entirely missing numeric/categorical columns are rejected.

    Distances use unscaled numeric features and category codes. Ordinal codes
    impose an artificial order; one-hot encoding may be preferable for nominal
    labels. This helper fits on every call: use a fitted train-only sklearn
    Pipeline instead for model evaluation or inference.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not df.columns.is_unique:
        raise ValueError("DataFrame columns must be unique")
    if method not in ("ordinal", "onehot"):
        raise ValueError("method must be 'ordinal' or 'onehot'")
    if (
        isinstance(n_neighbors, bool)
        or not isinstance(n_neighbors, int)
        or n_neighbors < 1
    ):
        raise ValueError("n_neighbors must be a positive integer")
    excluded = list(exclude_columns) if exclude_columns is not None else []
    unknown = set(excluded).difference(df.columns)
    if unknown:
        raise ValueError(f"Unknown exclude_columns: {unknown}")

    working_df = df.drop(columns=excluded)
    result = df.copy()

    if working_df.empty:
        return result

    encoded_df, transformer, categorical_cols = encode_categories(
        working_df, method=method
    )

    empty_columns = [
        col
        for col in working_df
        if working_df[col].isna().all()
        and (col in categorical_cols or pd.api.types.is_numeric_dtype(working_df[col]))
    ]

    if empty_columns:
        raise ValueError(
            f"Cannot infer values for entirely missing columns: {empty_columns}"
        )

    if method == "onehot" and categorical_cols:
        # OneHotEncoder treats NaN as a category. Mark the entire encoded block
        # missing so KNN actually imputes it instead of retaining the NaN label.
        encoder = transformer.named_transformers_["categories"]
        start = 0
        for col, categories in zip(categorical_cols, encoder.categories_, strict=True):
            stop = start + len(categories)
            encoded_df.loc[working_df[col].isna(), encoded_df.columns[start:stop]] = (
                np.nan
            )
            start = stop

    numeric_cols = [
        col
        for col in encoded_df.columns
        if pd.api.types.is_numeric_dtype(encoded_df[col].dtype)
    ]
    imputed_df = encoded_df.copy()
    if numeric_cols:
        imputer = KNNImputer(n_neighbors=n_neighbors, keep_empty_features=True)
        matrix = imputer.fit_transform(encoded_df[numeric_cols].astype(float))
        for i, col in enumerate(numeric_cols):
            imputed_df[col] = matrix[:, i]

    decoded_df = decode_categories(imputed_df, transformer, categorical_cols)
    for col in working_df:
        if working_df[col].isna().any():
            result[col] = decoded_df[col]

    return result


def missing_variable_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return counts and percentages per column (zero for an empty frame)."""
    counts = df.isna().sum().to_numpy()
    return pd.DataFrame(
        {
            "variable": df.columns.to_list(),
            "n_missing": counts,
            "n_cases": len(df),
            "pct_missing": counts / len(df) * 100 if len(df) else 0.0,
        }
    )


def missing_variable_table(df: pd.DataFrame) -> pd.DataFrame:
    """Count variables grouped by their number of missing observations."""
    return (
        missing_variable_summary(df)
        .value_counts("n_missing")
        .reset_index()
        .rename(columns={"n_missing": "n_missing_in_variable", "count": "n_variables"})
        .assign(
            pct_variables=lambda df_1: df_1.n_variables / df_1.n_variables.sum() * 100
        )
        .sort_values("pct_variables", ascending=False)
    )


def missing_case_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return counts and percentages per row, using only input columns."""
    counts = df.isna().sum(axis=1).to_numpy()
    n_variables = df.shape[1]
    return pd.DataFrame(
        {
            "case": df.index.to_list(),
            "n_missing": counts,
            "n_variables": n_variables,
            "pct_missing": counts / n_variables * 100 if n_variables else 0.0,
        },
        index=df.index,
    )


def missing_case_table(df: pd.DataFrame) -> pd.DataFrame:
    """Count cases grouped by their number of missing variables."""
    return (
        missing_case_summary(df)
        .value_counts("n_missing")
        .reset_index()
        .rename(columns={"n_missing": "n_missing_in_cases", "count": "n_cases"})
        .assign(pct_cases=lambda df_1: df_1.n_cases / df_1.n_cases.sum() * 100)
        .sort_values("pct_cases", ascending=False)
    )


def missing_variable_plot(df: pd.DataFrame):
    """Return an Axes with missing counts per variable."""

    df_plot = missing_variable_summary(df).sort_values("n_missing")

    plot_range = range(1, len(df_plot.index) + 1)

    plt.figure(figsize=(8, 10))
    plt.hlines(y=plot_range, xmin=0, xmax=df_plot.n_missing, color="black")
    plt.plot(df_plot.n_missing, plot_range, "o", color="black")
    plt.yticks(plot_range, df_plot.variable)
    plt.grid(axis="x")

    plt.xlabel("Number missing")
    plt.ylabel("Variable")
    return plt.gca()


def missing_case_plot(df: pd.DataFrame):
    """Return a FacetGrid with missing-count frequencies per row."""

    df_plot = missing_case_summary(df)

    grid = sns.displot(data=df_plot, x="n_missing", binwidth=1, color="black")

    plt.grid(axis="x")
    plt.xlabel("Number of missings in case")
    plt.ylabel("Number of cases")
    return grid


def missing_upsetplot(df: pd.DataFrame, variables: list[str] | None = None, **kwargs):
    """Plot intersections of missingness and return the UpSet axes dictionary."""
    if variables is None:
        variables = df.columns.tolist()

    return (
        df.isna()
        .value_counts(variables)
        .pipe(lambda df_plot: upsetplot.plot(df_plot, **kwargs))
    )
