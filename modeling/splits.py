"""Partición reproducible y agrupada por canal para la Etapa 2.

Este módulo es la única fuente de verdad sobre qué videos se usan para
entrenar, validar y probar:

* El conjunto de prueba (~20 %) es la columna ``split`` de
  ``data_processing/dataset_features_v1.csv``. Se generó con
  ``StratifiedGroupKFold`` (5 folds, semilla 42) agrupando por ``channel_id``.
* Dentro de train se generan 5 folds de validación cruzada con el mismo
  criterio (estratificados por clase y agrupados por canal) y se guardan en
  ``modeling/folds_v1.csv`` para que todos los modelos usen los mismos folds.

Uso desde otro script o notebook::

    from modeling.splits import load_train_test, predefined_cv

    X_train, y_train, X_test, y_test, meta_train, meta_test = load_train_test()
    cv = predefined_cv(meta_train)       # pasar a GridSearchCV / cross_validate

Ejecutar ``python modeling/splits.py`` regenera ``folds_v1.csv`` y el reporte
``modeling/reports/folds_report.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit, StratifiedGroupKFold

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = REPOSITORY_ROOT / "data_processing" / "dataset_features_v1.csv"
MODELING_DIR = Path(__file__).resolve().parent
FOLDS_PATH = MODELING_DIR / "folds_v1.csv"
REPORT_PATH = MODELING_DIR / "reports" / "folds_report.md"

TARGET = "classification_label"
METADATA_COLUMNS = ("video_id", "channel_id", "split")
N_FOLDS = 5
RANDOM_STATE = 42
CLASS_ORDER = ("disturbing", "irrelevant", "restricted", "suitable")


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    """Carga el dataset de features v1 sin convertir textos vacíos en NaN."""
    data = pd.read_csv(path, keep_default_na=False)
    missing = {TARGET, *METADATA_COLUMNS}.difference(data.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas {sorted(missing)} en {path}. "
            "Regenera el dataset con data_processing/preprocessing.py."
        )
    return data


def make_folds(data: pd.DataFrame) -> pd.DataFrame:
    """Asigna un fold de validación (0-4) a cada video de train; test = -1."""
    train = data.loc[data["split"].eq("train")]
    splitter = StratifiedGroupKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    fold = pd.Series(-1, index=data.index, dtype="int64")
    for fold_id, (_, val_idx) in enumerate(
        splitter.split(train, train[TARGET], groups=train["channel_id"])
    ):
        fold.loc[train.index[val_idx]] = fold_id

    folds = data.loc[:, ["video_id", "channel_id", "split", TARGET]].copy()
    folds["fold"] = fold
    _check_folds(folds)
    return folds


def _check_folds(folds: pd.DataFrame) -> None:
    """Verifica que no haya canales compartidos entre test y train ni entre folds."""
    channel_sets = folds.groupby("channel_id")["fold"].nunique()
    if channel_sets.gt(1).any():
        shared = channel_sets[channel_sets.gt(1)].index[:5].tolist()
        raise AssertionError(f"Canales repartidos en varios folds o en test: {shared}")
    if folds.loc[folds["split"].eq("train"), "fold"].lt(0).any():
        raise AssertionError("Hay videos de train sin fold asignado.")
    if folds.loc[folds["split"].eq("test"), "fold"].ne(-1).any():
        raise AssertionError("Hay videos de test con fold de validación.")


def load_folds(path: Path = FOLDS_PATH) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def load_train_test(
    features_path: Path = FEATURES_PATH, folds_path: Path = FOLDS_PATH
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Devuelve X/y de train y test, más los metadatos (incluye ``fold``).

    ``X`` no contiene ``video_id``, ``channel_id``, ``split`` ni la etiqueta.
    """
    data = load_features(features_path)
    folds = load_folds(folds_path).loc[:, ["video_id", "fold"]]
    data = data.merge(folds, on="video_id", how="left", validate="one_to_one")
    if data["fold"].isna().any():
        raise ValueError("folds_v1.csv no coincide con el dataset de features.")
    data["fold"] = data["fold"].astype("int64")

    meta_columns = [*METADATA_COLUMNS, "fold"]
    feature_columns = [c for c in data.columns if c not in {*meta_columns, TARGET}]
    is_train = data["split"].eq("train")
    train, test = data.loc[is_train].reset_index(drop=True), data.loc[~is_train].reset_index(drop=True)
    return (
        train[feature_columns],
        train[TARGET],
        test[feature_columns],
        test[TARGET],
        train[meta_columns],
        test[meta_columns],
    )


def predefined_cv(meta_train: pd.DataFrame) -> PredefinedSplit:
    """Validación cruzada con los folds fijos, para GridSearchCV o cross_validate."""
    return PredefinedSplit(test_fold=meta_train["fold"].to_numpy())


def build_report(folds: pd.DataFrame) -> str:
    """Tabla en Markdown con videos, canales y clases por fold y en test."""
    folds = folds.copy()
    folds["particion"] = np.where(
        folds["split"].eq("test"), "test", "fold " + folds["fold"].astype(str)
    )
    counts = pd.crosstab(folds["particion"], folds[TARGET]).reindex(
        columns=list(CLASS_ORDER), fill_value=0
    )
    shares = counts.div(counts.sum(axis=1), axis=0).mul(100)
    summary = pd.DataFrame(
        {
            "videos": counts.sum(axis=1),
            "canales": folds.groupby("particion")["channel_id"].nunique(),
        }
    )
    total = folds[TARGET].value_counts(normalize=True).mul(100)

    lines = [
        "# Reporte de la partición agrupada por canal (v1)",
        "",
        f"Generado por `modeling/splits.py` con `StratifiedGroupKFold` "
        f"({N_FOLDS} folds, semilla {RANDOM_STATE}), agrupando por `channel_id`.",
        "El test es la columna `split` de `dataset_features_v1.csv`; los folds se "
        "calculan solo dentro de train.",
        "",
        "| Partición | Videos | Canales | " + " | ".join(f"% {c}" for c in CLASS_ORDER) + " |",
        "|---|---:|---:|" + "---:|" * len(CLASS_ORDER),
    ]
    for name in summary.index:
        pct = " | ".join(f"{shares.loc[name, c]:.1f}" for c in CLASS_ORDER)
        lines.append(
            f"| {name} | {summary.loc[name, 'videos']:,} | "
            f"{summary.loc[name, 'canales']:,} | {pct} |"
        )
    pct_total = " | ".join(f"{total.get(c, 0):.1f}" for c in CLASS_ORDER)
    lines.append(
        f"| **ground truth** | {len(folds):,} | {folds['channel_id'].nunique():,} | {pct_total} |"
    )
    max_dev = (shares - total.reindex(list(CLASS_ORDER)).values).abs().max().max()
    lines += [
        "",
        f"- Desviación máxima de la proporción de una clase respecto al total: "
        f"{max_dev:.1f} puntos porcentuales.",
        "- Ningún canal aparece en más de una partición (verificado por `_check_folds`).",
        "- Nota: las imputaciones de `dataset_features_v1.csv` se ajustaron con todo "
        "train; el efecto dentro de la validación cruzada es mínimo (medianas), pero "
        "las transformaciones que aprenden de los datos (TF-IDF, escalado, SMOTE) "
        "deben ir dentro de un `Pipeline` para ajustarse solo con los folds de "
        "entrenamiento.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    folds = make_folds(load_features())
    FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    folds.drop(columns=[TARGET]).to_csv(FOLDS_PATH, index=False)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(folds)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"Folds guardados en {FOLDS_PATH}")


if __name__ == "__main__":
    main()
