"""Ajuste de hiperparámetros y selección del modelo final (Etapa 2).

1. ``RandomizedSearchCV`` con los folds agrupados por canal (``folds_v1``)
   sobre los dos mejores modelos de ``modeling/models.py``:

   * Regresión logística combinada (TF-IDF + metadatos).
   * LightGBM combinado (TF-IDF reducido con SVD + metadatos).

   Criterio de selección: F1 macro medio en validación cruzada.

2. El mejor de los dos se reentrena con todo train y se evalúa **una sola
   vez** en el conjunto de prueba (20 %, canales no vistos).

Salidas:

* ``modeling/reports/tuning.md``: espacio de búsqueda, mejores
  hiperparámetros y resultado en CV de cada modelo.
* ``modeling/reports/final_model.md``: métricas del modelo final en test.
* ``modeling/reports/test_predictions.csv``: predicción y probabilidades por
  video de test (para la matriz de confusión, curvas ROC/PR y análisis de
  errores).
* ``modeling/models/final.joblib``: pipeline completo (no se versiona). Es el
  que usa ``python -m modeling.additional_inference --model ...``.

Uso::

    python -m modeling.tuning              # búsqueda completa (20 combinaciones por modelo)
    python -m modeling.tuning --n-iter 5   # prueba rápida
"""

from __future__ import annotations

import argparse
import shutil
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from modeling.baselines import REPORTS_DIR, append_log, scoring
from modeling.features import build_preprocessor
from modeling.models import RANDOM_STATE, SVD_COMPONENTS, lightgbm
from modeling.splits import CLASS_ORDER, MODELING_DIR, load_train_test, predefined_cv

MODELS_DIR = MODELING_DIR / "models"
CACHE_DIR = MODELING_DIR / "outputs" / "cache"
FINAL_MODEL_PATH = MODELS_DIR / "final.joblib"
TUNING_REPORT = REPORTS_DIR / "tuning.md"
FINAL_REPORT = REPORTS_DIR / "final_model.md"
PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.csv"

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def search_spaces() -> dict[str, tuple[str, Pipeline, dict]]:
    """Nombre -> (vista, pipeline base, distribuciones de hiperparámetros).

    El paso ``features`` no cambia entre combinaciones, así que se cachea
    (``memory``) para no recalcular TF-IDF y SVD en cada una.
    """
    lr = Pipeline(
        [
            ("features", build_preprocessor("combined")),
            ("model", LogisticRegression(class_weight="balanced", max_iter=5000)),
        ],
        memory=str(CACHE_DIR),
    )
    lgbm = Pipeline(
        [
            (
                "features",
                build_preprocessor(
                    "combined", svd_components=SVD_COMPONENTS, random_state=RANDOM_STATE
                ),
            ),
            ("model", lightgbm()),
        ],
        memory=str(CACHE_DIR),
    )
    return {
        "LogReg combinado (TF-IDF + metadatos)": (
            "combined",
            lr,
            {"model__C": loguniform(0.05, 20)},
        ),
        f"LightGBM combinado (SVD {SVD_COMPONENTS} + metadatos)": (
            "combined",
            lgbm,
            {
                "model__n_estimators": randint(200, 1000),
                "model__learning_rate": loguniform(0.01, 0.2),
                "model__num_leaves": randint(8, 64),
                "model__min_child_samples": randint(5, 60),
                "model__colsample_bytree": uniform(0.4, 0.6),
                "model__reg_lambda": loguniform(1e-3, 10),
            },
        ),
    }


def run_search(name: str, pipeline: Pipeline, space: dict, X, y, cv, n_iter: int):
    print(f"Buscando hiperparámetros: {name} ({n_iter} combinaciones x 5 folds)...", flush=True)
    start = time.time()
    search = RandomizedSearchCV(
        pipeline,
        space,
        n_iter=n_iter,
        scoring=scoring(),
        refit="f1_macro",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    search.fit(X, y)
    print(f"  listo en {time.time() - start:.0f} s, F1 macro {search.best_score_:.3f}", flush=True)
    return search


def best_cv_metrics(search) -> dict[str, tuple[float, float]]:
    i = search.best_index_
    cvr = search.cv_results_
    return {
        name: (float(cvr[f"mean_test_{name}"][i]), float(cvr[f"std_test_{name}"][i]))
        for name in scoring()
    }


def fmt_params(params: dict) -> str:
    return ", ".join(
        f"`{k.removeprefix('model__')}={v:.4g}`" if isinstance(v, float) else f"`{k.removeprefix('model__')}={v}`"
        for k, v in sorted(params.items())
    )


SPACE_DESCRIPTIONS = {
    "LogReg combinado (TF-IDF + metadatos)": "`C` ∈ [0.05, 20] (log-uniforme)",
    f"LightGBM combinado (SVD {SVD_COMPONENTS} + metadatos)": (
        "`n_estimators` ∈ [200, 1000]; `learning_rate` ∈ [0.01, 0.2] (log); "
        "`num_leaves` ∈ [8, 64]; `min_child_samples` ∈ [5, 60]; "
        "`colsample_bytree` ∈ [0.4, 1.0]; `reg_lambda` ∈ [0.001, 10] (log)"
    ),
}


def tuning_report(searches: dict, chosen: str) -> str:
    lines = [
        "# Ajuste de hiperparámetros (v1)",
        "",
        "Método: `RandomizedSearchCV` con los 5 folds agrupados por canal "
        "(`folds_v1.csv`), solo sobre train. Criterio de selección: F1 macro medio en CV.",
        "",
        "| Modelo | Espacio de búsqueda | Combinaciones | Mejores hiperparámetros | F1 macro CV | Recall restricted CV |",
        "|---|---|---:|---|---:|---:|",
    ]
    for name, search in searches.items():
        m = best_cv_metrics(search)
        lines.append(
            f"| {name} | {SPACE_DESCRIPTIONS[name]} | {len(search.cv_results_['params'])} | "
            f"{fmt_params(search.best_params_)} | {m['f1_macro'][0]:.3f} ± {m['f1_macro'][1]:.3f} | "
            f"{m['recall_restricted'][0]:.3f} |"
        )
    lines += ["", f"**Modelo seleccionado:** {chosen}.", ""]
    return "\n".join(lines)


def evaluate_test(model, X_test, y_test, meta_test) -> tuple[str, pd.DataFrame]:
    """Métricas del modelo final en test y predicciones por video."""
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    classes = list(model.classes_)

    predictions = pd.DataFrame(
        {"video_id": meta_test["video_id"].to_numpy(), "y_true": y_test.to_numpy(), "y_pred": y_pred}
    )
    for i, label in enumerate(classes):
        predictions[f"proba_{label}"] = proba[:, i].round(5)

    labels = list(CLASS_ORDER)
    report = classification_report(y_test, y_pred, labels=labels, digits=3, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    lines = [
        "| Métrica | Valor |",
        "|---|---:|",
        f"| F1 macro | {f1_score(y_test, y_pred, average='macro'):.3f} |",
        f"| Recall macro | {recall_score(y_test, y_pred, average='macro'):.3f} |",
        f"| Accuracy | {report['accuracy']:.3f} |",
        "",
        "| Clase | Precision | Recall | F1 | Videos |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in labels:
        r = report[label]
        lines.append(
            f"| {label} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {int(r['support'])} |"
        )
    lines += [
        "",
        "Matriz de confusión (filas = clase real, columnas = predicción):",
        "",
        "| Real \\ Predicho | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
    ]
    for label, row in zip(labels, cm):
        lines.append(f"| {label} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines), predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-iter", type=int, default=20, help="combinaciones por modelo")
    args = parser.parse_args()

    X_train, y_train, X_test, y_test, meta_train, meta_test = load_train_test()
    cv = predefined_cv(meta_train)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    spaces = search_spaces()
    searches = {
        name: run_search(name, pipe, space, X_train, y_train, cv, args.n_iter)
        for name, (_, pipe, space) in spaces.items()
    }
    shutil.rmtree(CACHE_DIR, ignore_errors=True)

    chosen = max(searches, key=lambda n: searches[n].best_score_)
    final = searches[chosen].best_estimator_  # ya reentrenado con todo train
    final.set_params(memory=None)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TUNING_REPORT.write_text(tuning_report(searches, chosen), encoding="utf-8")
    append_log(
        {f"{n} (ajustado)": best_cv_metrics(s) for n, s in searches.items()},
        {f"{n} (ajustado)": spaces[n][0] for n in searches},
        etapa="tuning",
    )

    # Única evaluación en test, con el modelo ya elegido por CV.
    print(f"Evaluando en test el modelo final: {chosen}", flush=True)
    metrics, predictions = evaluate_test(final, X_test, y_test, meta_test)
    final_report = "\n".join(
        [
            "# Modelo final en el conjunto de prueba (v1)",
            "",
            f"Modelo: **{chosen}**, con {fmt_params(searches[chosen].best_params_)}.",
            "Elegido por F1 macro en validación cruzada (ver `tuning.md`), reentrenado con "
            f"todo train ({len(X_train):,} videos) y evaluado una sola vez en test "
            f"({len(X_test):,} videos de canales no vistos en entrenamiento).",
            "",
            metrics,
            "",
        ]
    )
    FINAL_REPORT.write_text(final_report, encoding="utf-8")
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, FINAL_MODEL_PATH)

    print(TUNING_REPORT.read_text(encoding="utf-8"))
    print(final_report)
    print(f"Modelo guardado en {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
