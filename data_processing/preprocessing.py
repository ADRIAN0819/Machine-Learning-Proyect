"""Feature preparation for the ground-truth YouTube dataset."""

import json
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
TITLE_STAT_COLUMNS = ("title_chars", "title_words")
MODEL_FEATURE_COLUMNS = (
    *TEXT_COLUMNS,
    *CATEGORICAL_COLUMNS,
    *TITLE_STAT_COLUMNS,
    "duration_seconds",
    *(f"{column}_log1p" for column in COUNT_COLUMNS),
    *RATIO_COLUMNS,
)
IMPUTATION_FILENAME = "imputation_v1.json"
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


def _parse_numeric(data: pd.DataFrame, strict: bool) -> pd.DataFrame:
    """Convierte duración y conteos a float64 y calcula las razones por vista.

    Con ``strict=True`` (ground truth) un valor no numérico o negativo es un
    error; con ``strict=False`` (datos sin etiqueta) se trata como faltante.
    """
    raw_numeric = pd.DataFrame(index=data.index)
    for column in ("duration_seconds", *COUNT_COLUMNS):
        source_values = data[column].astype("string").fillna("").str.strip()
        is_missing = source_values.eq("")
        # float64 explícito: con pandas >= 3 to_numeric devuelve Int64 y la
        # imputación con medianas no enteras (p. ej. 507.5) fallaría.
        converted = pd.to_numeric(
            source_values.mask(is_missing), errors="coerce"
        ).astype("float64")
        invalid_values = (~is_missing & converted.isna()) | converted.lt(0)
        if strict and invalid_values.any():
            raise ValueError(f"{column} contiene valores inválidos o negativos.")
        raw_numeric[column] = converted.mask(invalid_values)

    views = raw_numeric["view_count"]
    raw_numeric["likes_per_view"] = (
        raw_numeric["like_count"].div(views).where(views.gt(0))
    )
    raw_numeric["comments_per_view"] = (
        raw_numeric["comment_count"].div(views).where(views.gt(0))
    )
    return raw_numeric


def _normalize_categorical(values: pd.Series) -> pd.Series:
    values = values.astype("string").fillna("").str.strip().str.lower()
    return values.mask(values.eq(""))


def _assemble_features(
    data: pd.DataFrame,
    raw_numeric: pd.DataFrame,
    numeric_imputation: dict[str, float],
    categorical_imputation: dict[str, str],
) -> pd.DataFrame:
    """Aplica limpieza de texto, imputación y log(1+x) con valores ya ajustados."""
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

    # Longitud del título: el EDA mostró títulos más largos en suitable y
    # disturbing (~60 caracteres) que en irrelevant y restricted (~45).
    features["title_chars"] = features["title"].str.len().astype("int64")
    features["title_words"] = (
        features["title"].str.split().str.len().fillna(0).astype("int64")
    )

    for column in CATEGORICAL_COLUMNS:
        features[column] = _normalize_categorical(data[column]).fillna(
            categorical_imputation[column]
        )

    features["duration_seconds"] = raw_numeric["duration_seconds"].fillna(
        numeric_imputation["duration_seconds"]
    )
    for column in COUNT_COLUMNS:
        imputed_counts = raw_numeric[column].fillna(numeric_imputation[column])
        features[f"{column}_log1p"] = np.log1p(imputed_counts)
    for column in RATIO_COLUMNS:
        features[column] = raw_numeric[column].fillna(numeric_imputation[column])
    return features


def transform_unlabeled(
    data: pd.DataFrame, imputation_path: Path | None = None
) -> pd.DataFrame:
    """Transforma videos nuevos (sin etiqueta) con la imputación ajustada en train.

    Devuelve ``video_id``, ``channel_id`` y las mismas features que
    ``dataset_features_v1.csv``, en el mismo orden.
    """
    imputation_path = imputation_path or DEFAULT_OUTPUT_PATH.with_name(
        IMPUTATION_FILENAME
    )
    imputation = json.loads(Path(imputation_path).read_text(encoding="utf-8"))
    data = data.reset_index(drop=True)
    raw_numeric = _parse_numeric(data, strict=False)
    features = _assemble_features(
        data, raw_numeric, imputation["numeric"], imputation["categorical"]
    )
    features.insert(0, "channel_id", data["channel_id"].astype("string"))
    features.insert(0, "video_id", data["video_id"].astype("string").str.strip())
    return features.loc[:, ["video_id", "channel_id", *MODEL_FEATURE_COLUMNS]]


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

    raw_numeric = _parse_numeric(data, strict=True)

    train_numeric = raw_numeric.iloc[train_indices]
    imputation_values: dict[str, float] = {}
    for column in NUMERIC_COLUMNS:
        median = train_numeric[column].median()
        if pd.isna(median):
            raise ValueError(
                f"No hay valores observados en train para imputar {column}."
            )
        imputation_values[column] = float(median)

    train_categories = data.iloc[train_indices]
    categorical_imputation: dict[str, str] = {}
    for column in CATEGORICAL_COLUMNS:
        mode = _normalize_categorical(train_categories[column]).mode(dropna=True)
        if mode.empty:
            raise ValueError(
                f"No hay valores observados en train para imputar {column}."
            )
        categorical_imputation[column] = str(mode.iloc[0])

    features = _assemble_features(
        data, raw_numeric, imputation_values, categorical_imputation
    )
    features["classification_label"] = labels
    split = np.full(len(data), "train", dtype=object)
    split[test_indices] = "test"
    features["split"] = split

    model_feature_columns = list(MODEL_FEATURE_COLUMNS)
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
    imputation_path = output_path.with_name(IMPUTATION_FILENAME)
    imputation_path.write_text(
        json.dumps(
            {"numeric": imputation_values, "categorical": categorical_imputation},
            indent=2,
        ),
        encoding="utf-8",
    )
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
        "imputation_path": str(imputation_path),
    }
    return features, audit
