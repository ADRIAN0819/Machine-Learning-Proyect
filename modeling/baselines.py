"""Modelos baseline de la Etapa 2 (Sección IV-A/B del informe).

Compara, con la misma validación cruzada agrupada por canal (``folds_v1``):

* ``Dummy``: clase mayoritaria, piso de referencia.
* ``LogReg + TF-IDF``: regresión logística con ``class_weight="balanced"``
  sobre título, descripción y tags.
* ``Naive Bayes``: Multinomial NB sobre el mismo TF-IDF (solo texto, porque
  NB no admite los valores negativos del escalado de metadatos).

Solo se usa train; el conjunto de prueba no se toca hasta elegir el modelo
final. Todas las transformaciones van dentro del ``Pipeline`` y se ajustan
por fold.

Ejecutar ``python -m modeling.baselines`` genera
``modeling/reports/baselines.md`` y agrega una fila por modelo a
``modeling/reports/experimentos.csv`` (log de experimentos).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, recall_score
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from modeling.features import build_preprocessor
from modeling.splits import CLASS_ORDER, MODELING_DIR, load_train_test, predefined_cv

REPORTS_DIR = MODELING_DIR / "reports"
REPORT_PATH = REPORTS_DIR / "baselines.md"
LOG_PATH = REPORTS_DIR / "experimentos.csv"
FOCUS_CLASSES = ("disturbing", "restricted")


def scoring() -> dict[str, object]:
    """F1/recall macro y por clase (recall de las clases críticas)."""
    metrics: dict[str, object] = {
        "f1_macro": "f1_macro",
        "recall_macro": "recall_macro",
        "accuracy": "accuracy",
    }
    for label in CLASS_ORDER:
        metrics[f"f1_{label}"] = make_scorer(
            f1_score, labels=[label], average="macro", zero_division=0
        )
    for label in FOCUS_CLASSES:
        metrics[f"recall_{label}"] = make_scorer(
            recall_score, labels=[label], average="macro", zero_division=0
        )
    return metrics


def baseline_models() -> dict[str, tuple[str, Pipeline]]:
    """Nombre -> (vista de features, pipeline sin ajustar)."""
    return {
        "Dummy (clase mayoritaria)": (
            "text",
            Pipeline(
                [
                    ("features", build_preprocessor("text")),
                    ("model", DummyClassifier(strategy="most_frequent")),
                ]
            ),
        ),
        "LogReg + TF-IDF": (
            "text",
            Pipeline(
                [
                    ("features", build_preprocessor("text")),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced", max_iter=2000, C=1.0
                        ),
                    ),
                ]
            ),
        ),
        "Naive Bayes Multinomial + TF-IDF": (
            "text",
            Pipeline(
                [
                    ("features", build_preprocessor("text")),
                    ("model", MultinomialNB(alpha=0.1)),
                ]
            ),
        ),
    }


def evaluate(model: Pipeline, X, y, cv) -> dict[str, tuple[float, float]]:
    """Media y desviación estándar de cada métrica en los 5 folds."""
    result = cross_validate(model, X, y, cv=cv, scoring=scoring(), n_jobs=-1)
    return {
        name.removeprefix("test_"): (float(np.mean(v)), float(np.std(v)))
        for name, v in result.items()
        if name.startswith("test_")
    }


def build_report(results: dict[str, dict[str, tuple[float, float]]]) -> str:
    def fmt(pair: tuple[float, float]) -> str:
        return f"{pair[0]:.3f} ± {pair[1]:.3f}"

    lines = [
        "# Modelos baseline (v1)",
        "",
        "Validación cruzada de 5 folds agrupada por canal (`folds_v1.csv`), "
        "solo sobre train (3,842 videos). Media ± desviación estándar entre folds.",
        "",
        "| Modelo | F1 macro | Recall macro | Recall disturbing | Recall restricted | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name} | {fmt(m['f1_macro'])} | {fmt(m['recall_macro'])} | "
            f"{fmt(m['recall_disturbing'])} | {fmt(m['recall_restricted'])} | "
            f"{fmt(m['accuracy'])} |"
        )
    lines += [
        "",
        "F1 por clase:",
        "",
        "| Modelo | " + " | ".join(CLASS_ORDER) + " |",
        "|---|" + "---:|" * len(CLASS_ORDER),
    ]
    for name, m in results.items():
        lines.append(
            f"| {name} | " + " | ".join(f"{m[f'f1_{c}'][0]:.3f}" for c in CLASS_ORDER) + " |"
        )
    lines += [
        "",
        "- No se reporta accuracy como métrica principal: con 40 % de `irrelevant`, "
        "el modelo trivial ya obtiene una accuracy alta sin detectar ningún video "
        "`disturbing` ni `restricted`.",
    ]
    return "\n".join(lines) + "\n"


def append_log(
    results: dict[str, dict[str, tuple[float, float]]],
    views: dict[str, str],
    etapa: str = "baseline",
) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    rows = []
    for name, m in results.items():
        row = {"fecha": timestamp, "etapa": etapa, "modelo": name, "vista": views[name]}
        row.update({k: round(v[0], 4) for k, v in m.items()})
        row["f1_macro_std"] = round(m["f1_macro"][1], 4)
        rows.append(row)
    log = pd.DataFrame(rows)
    if LOG_PATH.exists():
        log = pd.concat([pd.read_csv(LOG_PATH), log], ignore_index=True)
    log.to_csv(LOG_PATH, index=False)


def main() -> None:
    X_train, y_train, _, _, meta_train, _ = load_train_test()
    cv = predefined_cv(meta_train)

    results, views = {}, {}
    for name, (view, model) in baseline_models().items():
        print(f"Evaluando {name}...")
        results[name] = evaluate(model, X_train, y_train, cv)
        views[name] = view

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report(results)
    REPORT_PATH.write_text(report, encoding="utf-8")
    append_log(results, views)
    print(report)
    print(f"Reporte en {REPORT_PATH}\nLog en {LOG_PATH}")


if __name__ == "__main__":
    main()
