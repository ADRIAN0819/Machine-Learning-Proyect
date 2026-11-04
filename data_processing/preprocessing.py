"""Feature preparation for the ground-truth YouTube dataset."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = (
    REPOSITORY_ROOT
    / "dataset_disturbed_youtube _for_kids"
    / "dataset_groundtruth.csv"
)
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "dataset_features_v1.csv"

REQUIRED_COLUMNS = {
    "video_id",
    "channel_id",
    "title",
    "description",
    "tags",
    "category_id",
    "duration_seconds",
    "definition",
    "caption",
    "licensed_content",
    "view_count",
    "like_count",
    "dislike_count",
    "comment_count",
    "classification_label",
}
TEXT_COLUMNS = ("title", "description", "tags")
CATEGORICAL_COLUMNS = (
    "category_id",
    "definition",
    "caption",
    "licensed_content",
)
COUNT_COLUMNS = ("view_count", "like_count", "dislike_count", "comment_count")
RATIO_COLUMNS = ("likes_per_view", "comments_per_view")
NUMERIC_COLUMNS = ("duration_seconds", *COUNT_COLUMNS, *RATIO_COLUMNS)
METADATA_COLUMNS = ("video_id", "channel_id", "split")
LEAKAGE_AND_IDENTIFIER_COLUMNS = (
    "prediction",
    "is_ground_truth",
    "source_dataset",
    "video_id",
    "channel_id",
    "channel_title",
    "published_at",
    "duration_iso",
    "favorite_count",
)


def build_features(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create and save an imputed feature dataset with a channel-safe split."""
    data = pd.read_csv(input_path, keep_default_na=False).reset_index(drop=True)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing_columns)}")
    if (
        data["video_id"].astype("string").str.strip().eq("").any()
        or data["video_id"].duplicated().any()
    ):
        raise ValueError("video_id debe estar presente y ser único.")
    if data["channel_id"].astype("string").str.strip().eq("").any():
        raise ValueError("channel_id es requerido para separar por canal.")
    if data["classification_label"].astype("string").str.strip().eq("").any():
        raise ValueError("classification_label no puede tener valores faltantes.")

    labels = data["classification_label"].astype("string").str.strip()
    groups = data["channel_id"].astype("string")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_indices, test_indices = next(
        splitter.split(data, labels, groups=groups)
    )
    train_channels = set(groups.iloc[train_indices])
    test_channels = set(groups.iloc[test_indices])
    if train_channels.intersection(test_channels):
        raise AssertionError("El split separó videos del mismo canal.")

    raw_numeric = pd.DataFrame(index=data.index)
    numeric_source_columns = ("duration_seconds", *COUNT_COLUMNS)
    for column in numeric_source_columns:
        source_values = data[column].astype("string").str.strip()
        is_missing = source_values.eq("")
        # float64 explícito: con pandas >= 3 to_numeric devuelve Int64 y la
        # imputación con medianas no enteras (p. ej. 507.5) fallaría.
        converted = pd.to_numeric(
            source_values.mask(is_missing), errors="coerce"
        ).astype("float64")
        invalid_values = ~is_missing & converted.isna()
        if invalid_values.any():
            raise ValueError(f"{column} contiene valores numéricos inválidos.")
        if (converted.dropna() < 0).any():
            raise ValueError(f"{column} contiene valores negativos.")
        raw_numeric[column] = converted

    views = raw_numeric["view_count"]
    raw_numeric["likes_per_view"] = (
        raw_numeric["like_count"].div(views).where(views.gt(0))
    )
    raw_numeric["comments_per_view"] = (
        raw_numeric["comment_count"].div(views).where(views.gt(0))
    )

    train_numeric = raw_numeric.iloc[train_indices]
    imputation_values: dict[str, float] = {}
    for column in NUMERIC_COLUMNS:
        median = train_numeric[column].median()
        if pd.isna(median):
            raise ValueError(
                f"No hay valores observados en train para imputar {column}."
            )
        imputation_values[column] = float(median)

    features = pd.DataFrame(index=data.index)
    for column in TEXT_COLUMNS:
        text = data[column].astype("string").fillna("").str.strip()
        if column == "tags":
            # str.lower de Python: resultado idéntico en pandas 2 y 3.
            text = text.map(
                lambda value: "|".join(
                    tag.strip()
                    for tag in str(value).lower().split("|")
                    if tag.strip()
                )
            )
        features[column] = text

    train_categories = data.iloc[train_indices]
    categorical_imputation: dict[str, str] = {}
    for column in CATEGORICAL_COLUMNS:
        values = data[column].astype("string").str.strip().str.lower()
        values = values.mask(values.eq(""))
        mode = train_categories[column].astype("string").str.strip().str.lower()
        mode = mode.mask(mode.eq("")).mode(dropna=True)
        if mode.empty:
            raise ValueError(
                f"No hay valores observados en train para imputar {column}."
            )
        fill_value = str(mode.iloc[0])
        categorical_imputation[column] = fill_value
        features[column] = values.fillna(fill_value)

    features["duration_seconds"] = raw_numeric["duration_seconds"].fillna(
        imputation_values["duration_seconds"]
    )
    for column in COUNT_COLUMNS:
        imputed_counts = raw_numeric[column].fillna(imputation_values[column])
        features[f"{column}_log1p"] = np.log1p(imputed_counts)
    for column in RATIO_COLUMNS:
        features[column] = raw_numeric[column].fillna(imputation_values[column])

    features["classification_label"] = labels
    split = np.full(len(data), "train", dtype=object)
    split[test_indices] = "test"
    features["split"] = split

    model_feature_columns = [
        *TEXT_COLUMNS,
        *CATEGORICAL_COLUMNS,
        "duration_seconds",
        *(f"{column}_log1p" for column in COUNT_COLUMNS),
        *RATIO_COLUMNS,
    ]
    # video_id y channel_id se exportan solo como metadatos: channel_id es
    # necesario para la validación cruzada agrupada por canal dentro de train.
    # Ninguna columna de METADATA_COLUMNS debe entrar al modelo.
    features["video_id"] = data["video_id"].astype("string").str.strip()
    features["channel_id"] = groups
    exported_columns = [
        *METADATA_COLUMNS,
        *model_feature_columns,
        "classification_label",
    ]
    features = features.loc[:, exported_columns]

    if features.isna().any().any():
        raise AssertionError("El dataset exportado todavía contiene faltantes.")
    numeric_features = features.select_dtypes(include="number")
    if not np.isfinite(
        numeric_features.to_numpy(dtype=np.float64, na_value=np.nan)
    ).all():
        raise AssertionError("Las features numéricas deben ser finitas.")
    if features["split"].value_counts().get("test", 0) == 0:
        raise AssertionError("El fold de test quedó vacío.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    # Con keep_default_na=False los faltantes llegan como cadenas vacías,
    # por eso se cuentan así en todas las columnas (incluidas las numéricas).
    missing_before = data.apply(
        lambda column: column.astype("string").fillna("").str.strip().eq("").sum()
    )
    audit = {
        "rows": len(features),
        "class_counts": labels.value_counts().to_dict(),
        "split_counts": features["split"].value_counts().to_dict(),
        "missing_before": missing_before.loc[lambda values: values.gt(0)].to_dict(),
        "numeric_imputation": imputation_values,
        "categorical_imputation": categorical_imputation,
        "metadata_columns": list(METADATA_COLUMNS),
        "model_feature_columns": model_feature_columns,
        "excluded_columns_present": sorted(
            set(LEAKAGE_AND_IDENTIFIER_COLUMNS).intersection(data.columns)
        ),
        "output_path": str(output_path),
    }
    return features, audit
