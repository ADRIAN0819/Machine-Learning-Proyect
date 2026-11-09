"""Evaluación completa del modelo final — Sección V y VI-A del informe.

Genera para el paper:

* Curvas ROC multiclase (one-vs-rest) con AUC por clase y AUC macro.
* Curvas Precision-Recall con Average Precision por clase y AP macro.
* Tabla de métricas completa (F1 macro, F1/recall/precision por clase).
* Reporte Markdown ``modeling/reports/evaluation.md``.

Las figuras se guardan en ``modeling/reports/figures/`` como PNG de alta
resolución, listas para incluir en el paper IEEE.

Uso::

    python -m modeling.evaluation

El script carga automáticamente ``modeling/models/final.joblib``. Si no
existe lo re-entrena con el pipeline del modelo ganador (LogReg combinado
TF-IDF + metadatos, C=1.16).

Salidas:
    modeling/reports/evaluation.md
    modeling/reports/figures/roc_curves.png
    modeling/reports/figures/pr_curves.png
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # no necesita display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

from modeling.features import build_preprocessor
from modeling.splits import CLASS_ORDER, MODELING_DIR, load_train_test

warnings.filterwarnings("ignore", message="X does not have valid feature names")

REPORTS_DIR = MODELING_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = MODELING_DIR / "models"
FINAL_MODEL_PATH = MODELS_DIR / "final.joblib"
PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.csv"
EVAL_REPORT = REPORTS_DIR / "evaluation.md"

# Paleta de colores por clase (consistente con el EDA del proyecto)
CLASS_COLORS = {
    "disturbing": "#e05c57",
    "irrelevant": "#5b8dd9",
    "restricted": "#f5a623",
    "suitable": "#4caf7d",
}
LABELS = list(CLASS_ORDER)


# ---------------------------------------------------------------------------
# Carga o re-entrenamiento del modelo
# ---------------------------------------------------------------------------

def _build_final_pipeline() -> Pipeline:
    """Reconstruye el pipeline ganador (LogReg combinado, C=1.16)."""
    return Pipeline(
        [
            ("features", build_preprocessor("combined")),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    C=1.16,
                    random_state=42,
                ),
            ),
        ]
    )


def load_or_train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Carga final.joblib si existe; si no, re-entrena y lo guarda."""
    if FINAL_MODEL_PATH.exists():
        print(f"Cargando modelo desde {FINAL_MODEL_PATH}...")
        return joblib.load(FINAL_MODEL_PATH)
    print("final.joblib no encontrado — re-entrenando LogReg combinado (C=1.16)...")
    model = _build_final_pipeline()
    model.fit(X_train, y_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, FINAL_MODEL_PATH)
    print(f"Modelo guardado en {FINAL_MODEL_PATH}")
    return model


# ---------------------------------------------------------------------------
# Curvas ROC
# ---------------------------------------------------------------------------

def plot_roc_curves(y_true: pd.Series, y_proba: np.ndarray, classes: list[str]) -> plt.Figure:
    """Curvas ROC one-vs-rest con AUC por clase y macro."""
    y_bin = label_binarize(y_true, classes=classes)

    fig, ax = plt.subplots(figsize=(7, 6))

    aucs = {}
    for i, label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        aucs[label] = roc_auc
        ax.plot(
            fpr, tpr,
            color=CLASS_COLORS[label],
            lw=2,
            label=f"{label}  (AUC = {roc_auc:.3f})",
        )

    macro_auc = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatorio (AUC = 0.500)")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.02])
    ax.set_xlabel("Tasa de Falsos Positivos", fontsize=12)
    ax.set_ylabel("Tasa de Verdaderos Positivos", fontsize=12)
    ax.set_title(f"Curvas ROC — AUC macro = {macro_auc:.3f}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, aucs, macro_auc


# ---------------------------------------------------------------------------
# Curvas Precision-Recall
# ---------------------------------------------------------------------------

def plot_pr_curves(y_true: pd.Series, y_proba: np.ndarray, classes: list[str]) -> plt.Figure:
    """Curvas Precision-Recall con Average Precision por clase y macro."""
    y_bin = label_binarize(y_true, classes=classes)

    fig, ax = plt.subplots(figsize=(7, 6))

    aps = {}
    for i, label in enumerate(classes):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_bin[:, i], y_proba[:, i])
        aps[label] = ap
        ax.plot(
            rec, prec,
            color=CLASS_COLORS[label],
            lw=2,
            label=f"{label}  (AP = {ap:.3f})",
        )

    macro_ap = average_precision_score(y_bin, y_proba, average="macro")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.02])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precisión", fontsize=12)
    ax.set_title(f"Curvas Precision-Recall — AP macro = {macro_ap:.3f}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, aps, macro_ap


# ---------------------------------------------------------------------------
# Matriz de confusión
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, classes: list[str]) -> plt.Figure:
    """Matriz de confusión normalizada por fila (recall por celda)."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Una sola matriz normalizada, con el conteo absoluto entre paréntesis.
    # Tamaño pensado para el ancho de una columna IEEE (~3.5 in).
    fig, ax = plt.subplots(figsize=(2.9, 2.6))
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                    ha="center", va="center", fontsize=7.5, color=color)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=7.5, rotation=25, ha="right")
    ax.set_yticklabels(classes, fontsize=7.5)
    ax.set_xlabel("Clase predicha", fontsize=8)
    ax.set_ylabel("Clase real", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Reporte Markdown
# ---------------------------------------------------------------------------

def build_eval_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: list[str],
    aucs: dict[str, float],
    macro_auc: float,
    aps: dict[str, float],
    macro_ap: float,
) -> str:
    report_dict = classification_report(
        y_true, y_pred, labels=classes, digits=3, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    lines = [
        "# Evaluación del modelo final en el conjunto de prueba",
        "",
        "Modelo: **LogReg combinado (TF-IDF + metadatos)**, `C=1.16`.",
        "Evaluado una sola vez sobre 955 videos de canales no vistos en entrenamiento.",
        "",
        "## Métricas globales",
        "",
        "| Métrica | Valor |",
        "|---|---:|",
        f"| F1 macro | {f1_score(y_true, y_pred, average='macro'):.3f} |",
        f"| Recall macro | {recall_score(y_true, y_pred, average='macro'):.3f} |",
        f"| Accuracy | {report_dict['accuracy']:.3f} |",
        f"| AUC macro (ROC, OvR) | {macro_auc:.3f} |",
        f"| AP macro (Prec-Recall) | {macro_ap:.3f} |",
        "",
        "## Métricas por clase",
        "",
        "| Clase | Precision | Recall | F1 | AUC | AP | Videos |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in classes:
        r = report_dict[label]
        lines.append(
            f"| {label} | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['f1-score']:.3f} | {aucs[label]:.3f} | {aps[label]:.3f} | "
            f"{int(r['support'])} |"
        )

    lines += [
        "",
        "## Matriz de confusión",
        "",
        "> Filas = clase real, columnas = predicción.",
        "",
        "| Real \\ Predicho | " + " | ".join(classes) + " |",
        "|---|" + "---:|" * len(classes),
    ]
    for label, row in zip(classes, cm):
        lines.append(f"| {label} | " + " | ".join(str(v) for v in row) + " |")

    lines += [
        "",
        "## Figuras generadas",
        "",
        "| Figura | Archivo |",
        "|---|---|",
        "| Curvas ROC | `reports/figures/roc_curves.png` |",
        "| Curvas Precision-Recall | `reports/figures/pr_curves.png` |",
        "| Matriz de confusión | `reports/figures/confusion_matrix.png` |",
        "",
        "## Notas metodológicas",
        "",
        "- Se usa **F1 macro** como métrica principal (no accuracy) porque el dataset",
        "  está desbalanceado: `irrelevant` representa el 40 % del total.",
        "  Con accuracy el modelo trivial (clase mayoritaria) ya obtendría ~41 %",
        "  sin detectar ningún video `disturbing` ni `restricted`.",
        "- Las curvas ROC y PR se calculan en modo **one-vs-rest** para cada clase.",
        "- `restricted` tiene el menor soporte (80 videos en test) y el AUC/AP más bajo,",
        "  lo que refleja la dificultad de distinguirla de `irrelevant`.",
        "- `class_weight='balanced'` en el modelo compensa el desbalance de clases",
        "  sin generar datos sintéticos (SMOTE), para evitar sobreajuste en train.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Cargando datos...")
    X_train, y_train, X_test, y_test, _, meta_test = load_train_test()

    model = load_or_train_model(X_train, y_train)

    print("Generando predicciones en test...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    classes = list(model.classes_)

    # Alinear el orden de clases con CLASS_ORDER
    col_order = [classes.index(c) for c in LABELS if c in classes]
    y_proba_ordered = y_proba[:, col_order]

    # Guardar/actualizar test_predictions.csv
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_df = pd.DataFrame({
        "video_id": meta_test["video_id"].to_numpy(),
        "y_true": y_test.to_numpy(),
        "y_pred": y_pred,
    })
    for i, label in enumerate(LABELS):
        predictions_df[f"proba_{label}"] = y_proba_ordered[:, i].round(5)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Predicciones guardadas en {PREDICTIONS_PATH}")

    # Curvas ROC
    print("Generando curvas ROC...")
    fig_roc, aucs, macro_auc = plot_roc_curves(y_test, y_proba_ordered, LABELS)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_roc.savefig(FIGURES_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig_roc)
    print(f"  AUC por clase: { {k: round(v,3) for k,v in aucs.items()} }")
    print(f"  AUC macro: {macro_auc:.3f}")

    # Curvas PR
    print("Generando curvas Precision-Recall...")
    fig_pr, aps, macro_ap = plot_pr_curves(y_test, y_proba_ordered, LABELS)
    fig_pr.savefig(FIGURES_DIR / "pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig_pr)
    print(f"  AP por clase: { {k: round(v,3) for k,v in aps.items()} }")
    print(f"  AP macro: {macro_ap:.3f}")

    # Matriz de confusión
    print("Generando matriz de confusión...")
    fig_cm = plot_confusion_matrix(y_test, y_pred, LABELS)
    fig_cm.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig_cm)

    # Reporte Markdown
    report = build_eval_report(
        y_test, y_pred, y_proba_ordered, LABELS, aucs, macro_auc, aps, macro_ap
    )
    EVAL_REPORT.write_text(report, encoding="utf-8")

    print(f"\n{'='*60}")
    print(report)
    print(f"Reporte guardado en {EVAL_REPORT}")
    print(f"Figuras guardadas en {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
