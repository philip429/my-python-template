import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def dispersion_measures_df(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize numeric columns using sample variance and 1.5-IQR outliers.

    Missing values are ignored. Columns with fewer than two observations have
    undefined (NaN) sample standard deviation and variance.
    """

    df_numeric = df.select_dtypes(include=["number"])

    if df_numeric.empty:
        return pd.DataFrame()

    q1 = df_numeric.quantile(0.25)
    q3 = df_numeric.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    dm_df = pd.DataFrame(index=df_numeric.columns).assign(
        mean=df_numeric.mean(),
        median=df_numeric.median(),
        range=df_numeric.max() - df_numeric.min(),
        IQR=iqr,
        Q1=q1,
        Q3=q3,
        std=df_numeric.std(),
        variance=df_numeric.var(),
        skewness=df_numeric.skew(),
        bottom_outlier=(df_numeric < lower_bound).sum(),
        top_outlier=(df_numeric > upper_bound).sum(),
    )

    return dm_df


# Keep the original misspelling working for notebooks already using it.
dispertion_measures_df = dispersion_measures_df


def encode_categories(
    df: pd.DataFrame, method: str = "ordinal"
) -> tuple[pd.DataFrame, ColumnTransformer, list[str]]:
    """Return (encoded_df, transformer, categorical_columns).

    Encode categories while preserving other supported columns and their
    dtypes: integers, floats, datetimes (including timezones), and timedeltas.
    Dates and durations pass through unchanged, including NaT. Generated
    category codes remain floats so ordinal encoding can represent NaN.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if not df.columns.is_unique:
        raise ValueError("DataFrame columns must be unique")
    if not all(isinstance(column, str) for column in df.columns):
        raise TypeError("DataFrame column names must be strings")

    if method not in ("ordinal", "onehot"):
        raise ValueError("method must be 'ordinal' or 'onehot'")

    categorical_cols = df.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    passthrough_cols = df.columns.difference(categorical_cols, sort=False)

    unsupported = [
        col
        for col in passthrough_cols
        if not (
            (
                pd.api.types.is_numeric_dtype(df[col].dtype)
                and not pd.api.types.is_complex_dtype(df[col].dtype)
            )
            or pd.api.types.is_datetime64_any_dtype(df[col].dtype)
            or pd.api.types.is_timedelta64_dtype(df[col].dtype)
        )
    ]

    if unsupported:
        raise TypeError(
            f"Unsupported columns: {unsupported}. Expected categorical, real numeric, "
            "datetime, or timedelta data; convert or remove unsupported columns."
        )

    prepared = df.copy()

    for col in categorical_cols:
        prepared[col] = (
            prepared[col].astype(object).where(prepared[col].notna(), np.nan)
        )

    encoder = (
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=np.nan,
        )
        if method == "ordinal"
        else OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    )

    transformer = ColumnTransformer(
        [("categories", encoder, categorical_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    # Pandas output keeps each passthrough column's dtype instead of coercing
    # dates, integers, and floats into a single NumPy matrix dtype.
    encoded_df = transformer.fit_transform(prepared)

    return encoded_df, transformer, categorical_cols


def decode_categories(
    encoded_df: pd.DataFrame,
    transformer: ColumnTransformer,
    categorical_cols: list[str],
) -> pd.DataFrame:
    """Restore labels and column order, but not original categorical dtypes.

    Fractional ordinal codes are rounded to the nearest valid category. This
    is an exploratory heuristic, not a model of categorical similarity.
    """

    if not categorical_cols:
        return encoded_df.loc[:, transformer.feature_names_in_].copy()

    encoder = transformer.named_transformers_["categories"]

    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    values = encoded_df[encoded_cols].to_numpy(dtype=float, copy=True)

    if isinstance(encoder, OrdinalEncoder):
        for j, categories in enumerate(encoder.categories_):
            # Exclude the missing-value category.
            n_categories = sum(not pd.isna(category) for category in categories)

            if n_categories == 0:
                values[:, j] = np.nan
                continue

            # Preserve unknown category codes.
            unknown_mask = values[:, j] == -1

            # Convert fractional KNN predictions to valid codes.
            values[:, j] = np.clip(
                np.rint(values[:, j]),
                0,
                n_categories - 1,
            )

            values[unknown_mask, j] = -1

    decoded = encoder.inverse_transform(values)

    categorical_df = pd.DataFrame(
        decoded,
        columns=categorical_cols,
        index=encoded_df.index,
    )

    numeric_df = encoded_df.drop(columns=encoded_cols)

    result = pd.concat(
        [categorical_df, numeric_df],
        axis=1,
    )

    return result.loc[:, transformer.feature_names_in_]


def draw_central_tendency_measures(
    values: pd.Series,
    ax: Axes,
    *,
    include_mode: bool = True,
    round_decimals: int | None = None,
    label_decimals: int = 2,
    max_categories: int = 12,
) -> None:
    """Draw appropriate central tendency measures by data type."""

    values = values.dropna()

    if values.empty:
        return

    # DATETIME
    if is_datetime64_any_dtype(values):
        median_date = values.median()

        ax.axvline(
            median_date,
            color="tab:green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_date:%Y-%m-%d}",
        )

        ax.set_xlabel("Date")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=9)

    # NUMERIC
    elif is_numeric_dtype(values) and not is_bool_dtype(values):
        measures = [
            ("Mean", values.mean(), "tab:red"),
            ("Median", values.median(), "tab:green"),
        ]

        if include_mode:
            mode_values = (
                values.round(round_decimals) if round_decimals is not None else values
            )

            counts = mode_values.value_counts()

            if (
                not counts.empty
                and counts.max() > 1
                and (counts == counts.max()).sum() == 1
            ):
                measures.append(("Mode", counts.idxmax(), "tab:blue"))

        for name, value, color in measures:
            ax.axvline(
                value,
                color=color,
                linestyle="--",
                linewidth=2,
                label=f"{name}: {value:.{label_decimals}f}",
            )

        ax.set_xlabel("Value")
        ax.legend(fontsize=9)

    # CATEGORICAL (str, object, category, bool)
    else:
        counts = values.value_counts(dropna=True)
        counts = counts[counts > 0]

        if counts.empty:
            return

        unique_mode = (
            include_mode and counts.max() > 1 and (counts == counts.max()).sum() == 1
        )

        mode = counts.idxmax() if unique_mode else None

        # Show most frequent categories, not all unique values.
        display_counts = counts.nlargest(max_categories)

        # Preserve ordering for ordered categorical columns.
        if isinstance(values.dtype, pd.CategoricalDtype) and values.dtype.ordered:
            display_counts = display_counts.sort_index()

        colors = [
            "tab:blue" if unique_mode and category == mode else "tab:gray"
            for category in display_counts.index
        ]

        ax.bar(
            range(len(display_counts)),
            display_counts.to_numpy(),
            color=colors,
        )

        ax.set_xticks(
            range(len(display_counts)),
            labels=[str(x) for x in display_counts.index],
            rotation=35,
            ha="right",
        )

        if len(counts) > max_categories:
            ax.set_xlabel(f"Category (top {max_categories} of {len(counts)})")
        else:
            ax.set_xlabel("Category")

        if unique_mode:
            percentage = counts.max() / counts.sum()

            ax.legend(
                handles=[
                    Patch(
                        facecolor="tab:blue",
                        label=f"Mode: {mode} ({percentage:.1%})",
                    )
                ],
                fontsize=9,
            )

    ax.set_ylabel("Count")


def plot_central_tendency_measures(
    data: pd.DataFrame,
    *,
    include_mode: bool = True,
    round_decimals: int | None = None,
    label_decimals: int = 2,
    bins: int = 10,
    col_wrap: int = 2,
    max_categories: int = 12,
    exclude_columns: list[str] | None = None,
    exclude_ids: bool = True,
    include_datetime: bool = True,
) -> sns.FacetGrid:
    """Plot numeric, categorical, and datetime distributions."""

    excluded = set(exclude_columns or [])
    selected_columns = []

    for column in data.columns:
        if column in excluded:
            continue

        # Automatically ignore common identifier names.
        normalized_name = str(column).strip().lower().replace(" ", "_")

        if exclude_ids and (
            normalized_name == "id"
            or normalized_name.endswith("_id")
            or normalized_name.startswith("id_")
        ):
            continue

        values = data[column].dropna()

        if values.empty:
            continue

        # Datetime columns
        if is_datetime64_any_dtype(values):
            if include_datetime and values.nunique() > 1:
                selected_columns.append(column)

        # Numeric columns
        elif is_numeric_dtype(values) and not is_bool_dtype(values):
            selected_columns.append(column)

        # Other columns: treat as categorical
        else:
            counts = values.value_counts(dropna=True)
            counts = counts[counts > 0]

            # Include useful repeated categorical values.
            if len(counts) >= 2 and counts.max() > 1:
                selected_columns.append(column)

    if not selected_columns:
        raise ValueError("No suitable columns found.")

    # Create a FacetGrid for every selected variable.
    grid = sns.FacetGrid(
        data=pd.DataFrame({"variable": selected_columns}),
        col="variable",
        col_wrap=col_wrap,
        height=4,
        aspect=1.4,
        sharex=False,
        sharey=False,
    )

    for column, ax in grid.axes_dict.items():
        values = data[column]

        is_numeric = is_numeric_dtype(values) and not is_bool_dtype(values)

        is_datetime = is_datetime64_any_dtype(values)

        # Only numeric and datetime variables get histograms.
        # Categorical bars are drawn by the helper.
        if is_numeric or is_datetime:
            sns.histplot(
                x=values.dropna(),
                bins=bins,
                ax=ax,
                color="lightsteelblue",
            )

        draw_central_tendency_measures(
            values=values,
            ax=ax,
            include_mode=include_mode,
            round_decimals=round_decimals,
            label_decimals=label_decimals,
            max_categories=max_categories,
        )

    grid.set_titles("{col_name}")
    grid.tight_layout()

    return grid
