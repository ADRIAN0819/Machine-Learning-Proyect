"""Modelos tabulares, combinado y ablation study de la Etapa 2.

Usa la misma validación cruzada agrupada por canal que los baselines
(``folds_v1``) y solo datos de train. Dos bloques:

1. Modelos comparados (Sección IV-B del informe):

   * Random Forest y LightGBM solo con metadatos.
   * LightGBM combinado: texto reducido con TF-IDF + ``TruncatedSVD`` y
     metadatos.
   * Regresión logística combinada: TF-IDF completo (disperso) y metadatos.

2. Ablation study: el mismo LightGBM entrenado con texto solo, metadatos
   solo y la combinación, para medir cuánto aporta cada fuente.

Todos los modelos usan ``class_weight`` balanceado para el desbalance de
clases (en lugar de SMOTE). Las transformaciones se ajustan dentro del
``Pipeline`` en cada fold.

Ejecutar ``python -m modeling.models`` genera ``modeling/reports/models.md``
y agrega filas a ``modeling/reports/experimentos.csv``.
"""

from __future__ import annotations

import warnings

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from modeling.baselines import REPORTS_DIR, append_log, evaluate
from modeling.features import build_preprocessor
from modeling.splits import CLASS_ORDER, load_train_test, predefined_cv

REPORT_PATH = REPORTS_DIR / "models.md"
SVD_COMPONENTS = 200
RANDOM_STATE = 42

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def lightgbm() -> LGBMClassifier:
    """LightGBM con hiperparámetros iniciales (sin ajustar todavía)."""
    return LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,  # la paralelización va por folds en cross_validate
        verbose=-1,
    )


def random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def pipeline(view: str, model, svd: bool = False) -> Pipeline:
    features = build_preprocessor(
        view, svd_components=SVD_COMPONENTS if svd else None, random_state=RANDOM_STATE
    )
    return Pipeline([("features", features), ("model", model)])


COMBINED_LGBM = f"LightGBM combinado (SVD {SVD_COMPONENTS} + metadatos)"


def compared_models() -> dict[str, tuple[str, Pipeline]]:
    """Nombre -> (vista, pipeline). Modelos de la Sección IV-B."""
    return {
        "Random Forest (metadatos)": ("metadata", pipeline("metadata", random_forest())),
        "LightGBM (metadatos)": ("metadata", pipeline("metadata", lightgbm())),
        COMBINED_LGBM: (
            "combined",
            pipeline("combined", lightgbm(), svd=True),
        ),
        "LogReg combinado (TF-IDF + metadatos)": (
            "combined",
            pipeline(
                "combined",
                LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0),
            ),
        ),
    }


def ablation_models() -> dict[str, tuple[str, Pipeline]]:
    """Mismo LightGBM con texto solo. Las filas "solo metadatos" y "texto +
    metadatos" son los mismos pipelines de ``compared_models``, así que se
    reutilizan sus resultados en lugar de entrenarlos dos veces."""
    return {
        f"Solo texto (SVD {SVD_COMPONENTS})": ("text", pipeline("text", lightgbm(), svd=True)),
    }


def fmt(pair: tuple[float, float]) -> str:
    return f"{pair[0]:.3f} ± {pair[1]:.3f}"


def results_table(results: dict[str, dict[str, tuple[float, float]]], first_col: str) -> list[str]:
    lines = [
        f"| {first_col} | F1 macro | Recall macro | Recall disturbing | Recall restricted | "
        + " | ".join(f"F1 {c}" for c in CLASS_ORDER)
        + " |",
        "|---|" + "---:|" * (4 + len(CLASS_ORDER)),
    ]
    for name, m in results.items():
        per_class = " | ".join(f"{m[f'f1_{c}'][0]:.3f}" for c in CLASS_ORDER)
        lines.append(
            f"| {name} | {fmt(m['f1_macro'])} | {fmt(m['recall_macro'])} | "
            f"{fmt(m['recall_disturbing'])} | {fmt(m['recall_restricted'])} | {per_class} |"
        )
    return lines


def build_report(compared, ablation) -> str:
    lines = [
        "# Modelos tabulares, combinado y ablation study (v1)",
        "",
        "Validación cruzada de 5 folds agrupada por canal (`folds_v1.csv`), solo "
        "sobre train (3,842 videos). Media ± desviación estándar entre folds. "
        "Hiperparámetros iniciales, todavía sin ajustar.",
        "",
        "## Modelos comparados",
        "",
        *results_table(compared, "Modelo"),
        "",
        "## Ablation study (LightGBM)",
        "",
        "Mismo modelo y mismos hiperparámetros; solo cambia la información de entrada.",
        "",
        *results_table(ablation, "Entrada"),
        "",
    ]
    return "\n".join(lines)


def run(models: dict[str, tuple[str, Pipeline]], X, y, cv) -> tuple[dict, dict]:
    results, views = {}, {}
    for name, (view, model) in models.items():
        print(f"Evaluando {name}...", flush=True)
        results[name] = evaluate(model, X, y, cv)
        views[name] = view
    return results, views


def main() -> None:
    X_train, y_train, _, _, meta_train, _ = load_train_test()
    cv = predefined_cv(meta_train)

    compared, compared_views = run(compared_models(), X_train, y_train, cv)
    text_only, text_views = run(ablation_models(), X_train, y_train, cv)
    ablation, ablation_views = dict(text_only), dict(text_views)
    # Las demás filas del ablation ya se evaluaron en compared_models.
    for name, source, view in (
        ("Solo metadatos", "LightGBM (metadatos)", "metadata"),
        ("Texto + metadatos", COMBINED_LGBM, "combined"),
    ):
        ablation[name] = compared[source]
        ablation_views[name] = view

    report = build_report(compared, ablation)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    append_log(compared, compared_views, etapa="modelos")
    append_log(text_only, text_views, etapa="ablation")
    print(report)
    print(f"Reporte en {REPORT_PATH}")


if __name__ == "__main__":
    main()
