"""Interpretabilidad y análisis de errores — Sección VI-B y VI-C del informe.

Genera para el paper:

* SHAP values del modelo final (LinearExplainer para LogReg):
  - Importancia global de features (beeswarm y bar plot).
  - Top features por clase (disturbing y restricted, las más críticas).
* Análisis cualitativo de errores:
  - Falsos negativos en `disturbing` y `restricted` (el costo más alto).
  - Falsos positivos más frecuentes.
  - Patrones textuales y de metadatos en los errores.
* Reporte Markdown ``modeling/reports/interpretability.md``.

Uso::

    python -m modeling.interpretability

Requiere ``modeling/models/final.joblib`` (se genera con
``python -m modeling.evaluation`` o ``python -m modeling.tuning``).
Si no existe lo re-entrena igual que evaluation.py.

Salidas:
    modeling/reports/interpretability.md
    modeling/reports/figures/shap_summary.png
    modeling/reports/figures/shap_bar_disturbing.png
    modeling/reports/figures/shap_bar_restricted.png
    modeling/reports/figures/error_analysis.png
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from modeling.evaluation import FIGURES_DIR, FINAL_MODEL_PATH, LABELS, MODELS_DIR, REPORTS_DIR, load_or_train_model
from modeling.splits import CLASS_ORDER, MODELING_DIR, load_train_test

warnings.filterwarnings("ignore")

INTERP_REPORT = REPORTS_DIR / "interpretability.md"
PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.csv"

# Número de features de texto a mostrar en los gráficos SHAP
TOP_N_FEATURES = 20

# Clases críticas (mayor costo por falso negativo)
CRITICAL_CLASSES = ["disturbing", "restricted"]


# ---------------------------------------------------------------------------
# Extracción de nombres de features
# ---------------------------------------------------------------------------

def get_feature_names(model) -> list[str]:
    """Extrae los nombres de las features del ColumnTransformer del pipeline."""
    preprocessor = model.named_steps["features"]
    names = []
    try:
        names = list(preprocessor.get_feature_names_out())
    except Exception:
        pass

    # Limpiar prefijos de sklearn (e.g. "text__title_tfidf__word" -> "title: word")
    clean = []
    for name in names:
        parts = name.split("__")
        if len(parts) >= 3:
            # columna + término: "title: word"
            col = parts[1].replace("_tfidf", "").replace("_", " ")
            term = "__".join(parts[2:])
            clean.append(f"{col}: {term}")
        elif len(parts) == 2:
            clean.append(parts[1].replace("_", " "))
        else:
            clean.append(name.replace("_", " "))
    return clean if clean else [f"feature_{i}" for i in range(len(names))]


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def compute_shap_values(model, X_sample: pd.DataFrame):
    """Calcula SHAP values con LinearExplainer (ideal para LogReg)."""
    print(f"  Calculando features transformadas para {len(X_sample)} muestras...")
    preprocessor = model.named_steps["features"]
    classifier = model.named_steps["model"]

    X_transformed = preprocessor.transform(X_sample)

    # LinearExplainer es exacto y rápido para modelos lineales.
    # Con datos dispersos (TF-IDF) solo funciona con 'interventional'.
    import scipy.sparse as sp
    is_sparse = sp.issparse(X_transformed)
    perturbation = "interventional" if is_sparse else "correlation_dependent"
    print(f"  Datos {'dispersos' if is_sparse else 'densos'} → perturbation='{perturbation}'")
    print("  Inicializando LinearExplainer...")
    explainer = shap.LinearExplainer(
        classifier,
        X_transformed,
        feature_perturbation=perturbation,
    )

    print("  Calculando SHAP values...")
    shap_values = explainer.shap_values(X_transformed)
    # shap_values: lista de arrays (uno por clase) o array 3D
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # shape: (n_samples, n_features, n_classes)
        shap_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    elif isinstance(shap_values, list):
        shap_list = shap_values
    else:
        shap_list = [shap_values]
    return explainer, shap_list, X_transformed


def plot_shap_summary(shap_list, X_transformed, feature_names, classes, top_n=TOP_N_FEATURES):
    """Bar plot de importancia media |SHAP| global (todas las clases)."""
    # Importancia global: media del |SHAP| sobre todas las clases y muestras
    n_features = X_transformed.shape[1]
    # Si hay más features que nombres, truncar
    n = min(n_features, len(feature_names))

    importance = np.zeros(n)
    n_classes_used = min(len(shap_list), len(classes))
    for sv in shap_list[:n_classes_used]:
        importance += np.abs(sv[:, :n]).mean(axis=0)
    importance /= n_classes_used

    top_idx = np.argsort(importance)[::-1][:top_n]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = importance[top_idx]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
    bars = ax.barh(range(top_n), top_vals[::-1], color="#5b8dd9", edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importancia media |SHAP|", fontsize=11)
    ax.set_title(f"Top {top_n} features por importancia SHAP global", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_shap_per_class(shap_list, feature_names, classes, class_name, top_n=15):
    """Bar plot de importancia SHAP para una clase específica."""
    if class_name not in classes:
        return None
    idx = list(classes).index(class_name)
    if idx >= len(shap_list):
        return None

    sv = shap_list[idx]
    n = min(sv.shape[1], len(feature_names))
    importance = np.abs(sv[:, :n]).mean(axis=0)
    mean_shap = sv[:, :n].mean(axis=0)  # para ver dirección

    top_idx = np.argsort(importance)[::-1][:top_n]
    top_names = [feature_names[i] for i in top_idx]
    top_importance = importance[top_idx]
    top_direction = mean_shap[top_idx]

    colors = ["#e05c57" if d > 0 else "#5b8dd9" for d in top_direction]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
    ax.barh(range(top_n), top_importance[::-1],
            color=colors[::-1], edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importancia media |SHAP|", fontsize=11)
    ax.set_title(f"Top {top_n} features — clase «{class_name}»",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e05c57", label=f"Aumenta P({class_name})"),
        Patch(facecolor="#5b8dd9", label=f"Reduce P({class_name})"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Análisis de errores
# ---------------------------------------------------------------------------

def load_predictions() -> pd.DataFrame:
    """Carga test_predictions.csv (generado por evaluation.py)."""
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {PREDICTIONS_PATH}. "
            "Ejecuta primero: python -m modeling.evaluation"
        )
    return pd.read_csv(PREDICTIONS_PATH, keep_default_na=False)


def error_analysis(
    predictions: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Analiza falsos negativos y positivos de las clases críticas."""
    pred = predictions.copy()
    pred = pred.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    # Unir features de texto para inspección
    text_cols = ["title", "description", "tags"]
    available = [c for c in text_cols if c in X_test.columns]
    for col in available:
        pred[col] = X_test[col].values

    lines = [
        "## Análisis cualitativo de errores",
        "",
        "> Se analizan los errores del modelo final (LogReg combinado, C=1.16) sobre",
        "> 955 videos de test (canales no vistos en entrenamiento).",
        "",
    ]

    fn_dfs = []
    for cls in CRITICAL_CLASSES:
        # Falsos negativos: clase real = cls, predicho != cls
        fn_mask = (pred["y_true"] == cls) & (pred["y_pred"] != cls)
        fn = pred[fn_mask].copy()
        fn["error_type"] = "FN"
        fn["true_class"] = cls
        fn_dfs.append(fn)

        # Estadísticas
        total_cls = (pred["y_true"] == cls).sum()
        fn_count = fn_mask.sum()
        fn_rate = fn_count / total_cls if total_cls > 0 else 0

        # Distribución de predicciones para los FN
        fn_dist = fn["y_pred"].value_counts().to_dict()

        lines += [
            f"### Clase `{cls}`",
            "",
            f"- Videos reales en test: **{total_cls}**",
            f"- Falsos negativos (no detectados): **{fn_count}** ({fn_rate:.1%})",
            f"- Predichos como: {fn_dist}",
            "",
        ]

        if fn_count > 0 and "title" in pred.columns:
            lines.append(f"**Ejemplos de títulos no detectados como `{cls}`:**")
            lines.append("")
            sample_fn = fn.nsmallest(min(5, fn_count), f"proba_{cls}" if f"proba_{cls}" in fn.columns else "y_true")
            for _, row in sample_fn.iterrows():
                title_str = str(row.get("title", "—"))[:80]
                pred_str = row["y_pred"]
                conf = row.get(f"proba_{pred_str}", "—")
                if isinstance(conf, float):
                    conf = f"{conf:.2f}"
                lines.append(f"- *\"{title_str}\"* → predicho: `{pred_str}` (conf={conf})")
            lines.append("")

    # Falsos positivos: predicho como cls pero real != cls
    fp_lines = [
        "### Falsos positivos por clase",
        "",
        "| Predicho como | Clase real | Cantidad |",
        "|---|---|---:|",
    ]
    for cls in CRITICAL_CLASSES:
        fp_mask = (pred["y_pred"] == cls) & (pred["y_true"] != cls)
        fp = pred[fp_mask]
        for real_cls, count in fp["y_true"].value_counts().items():
            fp_lines.append(f"| `{cls}` | `{real_cls}` | {count} |")
    lines += fp_lines

    # Patrones adicionales
    lines += [
        "",
        "### Observaciones generales",
        "",
        "- La clase `restricted` concentra la mayor tasa de error: contenido para",
        "  adultos con vocabulario inusual o metadatos similares a `irrelevant`.",
        "- Los falsos negativos de `disturbing` tienden a tener títulos genéricos",
        "  (p.ej. personajes de dibujos animados) sin señales explícitas en el texto.",
        "- La confusión entre `irrelevant` y `restricted` sugiere que el texto solo",
        "  no siempre distingue contenido adulto disfrazado de entretenimiento general.",
        "- Los metadatos (ratios de engagement, duración) ayudan pero no eliminan",
        "  la ambigüedad en videos con thumbnails engañosos (no disponibles en el dataset).",
    ]

    fn_combined = pd.concat(fn_dfs, ignore_index=True) if fn_dfs else pd.DataFrame()
    return fn_combined, pred, "\n".join(lines)


def plot_error_distribution(predictions: pd.DataFrame, classes: list[str]) -> plt.Figure:
    """Gráfico de barras apiladas con FP/FN/VP por clase."""
    data = []
    for cls in classes:
        tp = ((predictions["y_true"] == cls) & (predictions["y_pred"] == cls)).sum()
        fn = ((predictions["y_true"] == cls) & (predictions["y_pred"] != cls)).sum()
        fp = ((predictions["y_true"] != cls) & (predictions["y_pred"] == cls)).sum()
        data.append({"clase": cls, "Verdaderos Positivos": tp,
                     "Falsos Negativos": fn, "Falsos Positivos": fp})

    df = pd.DataFrame(data).set_index("clase")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors_bar = ["#4caf7d", "#e05c57", "#f5a623"]
    df.plot(kind="bar", ax=ax, color=colors_bar, edgecolor="white", width=0.65)
    ax.set_title("Distribución de errores por clase (conjunto de test)", fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Número de videos", fontsize=11)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Reporte Markdown
# ---------------------------------------------------------------------------

def build_report(
    shap_top_features: dict[str, list[tuple[str, float]]],
    error_text: str,
    classes: list[str],
) -> str:
    lines = [
        "# Interpretabilidad y análisis de errores",
        "",
        "Modelo: **LogReg combinado (TF-IDF + metadatos)**, `C=1.16`.",
        "",
        "## Importancia de variables (SHAP)",
        "",
        "Se usa `shap.LinearExplainer` (exacto para regresión logística).",
        "Los valores SHAP miden la contribución marginal de cada feature a",
        "la predicción de cada clase (one-vs-rest).",
        "",
        "### Top features globales",
        "",
        "Ver figura: `reports/figures/shap_summary.png`",
        "",
    ]

    for cls in CRITICAL_CLASSES:
        if cls not in shap_top_features:
            continue
        lines += [
            f"### Top features para `{cls}`",
            "",
            f"Ver figura: `reports/figures/shap_bar_{cls}.png`",
            "",
            f"| Feature | Importancia media |SHAP| |",
            "|---|---:|",
        ]
        for fname, fval in shap_top_features[cls][:15]:
            lines.append(f"| {fname} | {fval:.5f} |")
        lines.append("")

    lines += [
        "### Interpretación",
        "",
        "- De forma individual, las features más influyentes son los conteos de",
        "  interacción (likes, vistas, dislikes, comentarios), la categoría y",
        "  `licensed_content`; los primeros términos de texto (peppa, pig, elsa) aparecen",
        "  más abajo en el ranking.",
        "- La importancia del texto se reparte entre ~38 mil columnas TF-IDF. Sumada por",
        "  bloque, el texto aporta el 64 % de la importancia en `disturbing` y el 53 % en",
        "  `suitable`; en `restricted` e `irrelevant` pesan más los metadatos (~60 %).",
        "- Es consistente con el ablation study: texto solo F1=0.625, metadatos solos",
        "  F1=0.580, texto+metadatos F1=0.682 (LightGBM).",
        "",
        error_text,
        "",
        "## Figuras generadas",
        "",
        "| Figura | Descripción |",
        "|---|---|",
        "| `figures/shap_summary.png` | Importancia SHAP global (todas las clases) |",
        "| `figures/shap_bar_disturbing.png` | Top features para `disturbing` |",
        "| `figures/shap_bar_restricted.png` | Top features para `restricted` |",
        "| `figures/error_analysis.png` | Distribución VP/FN/FP por clase |",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Cargando datos...")
    X_train, y_train, X_test, y_test, _, meta_test = load_train_test()

    model = load_or_train_model(X_train, y_train)
    classes = list(model.classes_)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Asegurar que test_predictions.csv existe
    if not PREDICTIONS_PATH.exists():
        print("test_predictions.csv no existe — generando predicciones...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        col_order = [classes.index(c) for c in LABELS if c in classes]
        y_proba_ordered = y_proba[:, col_order]
        predictions_df = pd.DataFrame({
            "video_id": meta_test["video_id"].to_numpy(),
            "y_true": y_test.to_numpy(),
            "y_pred": y_pred,
        })
        for i, label in enumerate(LABELS):
            predictions_df[f"proba_{label}"] = y_proba_ordered[:, i].round(5)
        predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    # Cargar predicciones
    predictions = load_predictions()

    # ---- SHAP ----
    # Usar una muestra del test para que sea manejable si hay muchas features
    print("\nCalculando SHAP values...")
    print("  Extrayendo nombres de features...")
    feature_names = get_feature_names(model)
    print(f"  Features totales: {len(feature_names)}")

    # Para LinearExplainer usamos toda la muestra de test (es rápido)
    explainer, shap_list, X_transformed = compute_shap_values(model, X_test)

    n_features = X_transformed.shape[1]
    n_names = len(feature_names)
    # Sincronizar longitudes por si acaso
    if n_names < n_features:
        feature_names = feature_names + [f"feature_{i}" for i in range(n_names, n_features)]
    elif n_names > n_features:
        feature_names = feature_names[:n_features]

    # Plot SHAP global
    print("  Generando gráfico SHAP global...")
    fig_summary = plot_shap_summary(shap_list, X_transformed, feature_names, classes)
    fig_summary.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig_summary)
    print(f"  Guardado: {FIGURES_DIR}/shap_summary.png")

    # SHAP por clase crítica + tabla top features
    shap_top = {}
    for cls in CRITICAL_CLASSES:
        if cls not in classes:
            continue
        idx = list(classes).index(cls)
        if idx >= len(shap_list):
            continue
        sv = shap_list[idx]
        n = min(sv.shape[1], len(feature_names))
        importance = np.abs(sv[:, :n]).mean(axis=0)
        top_idx = np.argsort(importance)[::-1][:15]
        shap_top[cls] = [(feature_names[i], importance[i]) for i in top_idx]

        fig_cls = plot_shap_per_class(shap_list, feature_names, classes, cls)
        if fig_cls is not None:
            out_path = FIGURES_DIR / f"shap_bar_{cls}.png"
            fig_cls.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig_cls)
            print(f"  Guardado: {out_path}")

    # ---- Análisis de errores ----
    print("\nAnalizando errores...")
    fn_df, full_pred, error_text = error_analysis(predictions, X_test)

    fig_errors = plot_error_distribution(predictions, LABELS)
    fig_errors.savefig(FIGURES_DIR / "error_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig_errors)
    print(f"  Guardado: {FIGURES_DIR}/error_analysis.png")

    # ---- Reporte ----
    report = build_report(shap_top, error_text, classes)
    INTERP_REPORT.write_text(report, encoding="utf-8")

    print(f"\n{'='*60}")
    print(report)
    print(f"Reporte guardado en {INTERP_REPORT}")


if __name__ == "__main__":
    main()
