"""Comparación pareada fold a fold de los dos modelos ajustados (Etapa 2).

``tuning.py`` elige el modelo final por el F1 macro medio en CV, pero la
diferencia entre la regresión logística y LightGBM (0.710 vs. 0.698) es menor
que la desviación estándar entre folds. Como ambos se evaluaron con los mismos
5 folds agrupados por canal, la comparación correcta es pareada: diferencia
fold a fold y prueba t pareada.

Los hiperparámetros se reconstruyen exactamente con el mismo
``ParameterSampler`` (20 combinaciones, semilla 42) que usa
``RandomizedSearchCV`` en ``tuning.py``, tomando la combinación que coincide
con los mejores valores reportados en ``reports/tuning.md``.

Ejecutar ``python -m modeling.paired_comparison`` genera
``modeling/reports/paired_comparison.md``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.model_selection import ParameterSampler, cross_validate

from modeling.baselines import REPORTS_DIR, scoring
from modeling.models import RANDOM_STATE
from modeling.splits import load_train_test, predefined_cv
from modeling.tuning import search_spaces

REPORT_PATH = REPORTS_DIR / "paired_comparison.md"
N_ITER = 20  # igual que tuning.py

# Mejores valores de reports/tuning.md (redondeados), para ubicar la combinación exacta.
BEST_ROUNDED = {
    "LogReg combinado (TF-IDF + metadatos)": {"model__C": 1.16},
    "LightGBM combinado (SVD 200 + metadatos)": {
        "model__n_estimators": 227,
        "model__num_leaves": 14,
        "model__min_child_samples": 45,
    },
}
METRICS = ("f1_macro", "recall_restricted", "recall_disturbing")

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def best_params(name: str, space: dict) -> dict:
    """Combinación exacta muestreada por RandomizedSearchCV que coincide con tuning.md."""
    target = BEST_ROUNDED[name]
    matches = [
        params
        for params in ParameterSampler(space, N_ITER, random_state=RANDOM_STATE)
        if all(
            math.isclose(params[k], v, rel_tol=5e-3) if isinstance(v, float) else params[k] == v
            for k, v in target.items()
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{name}: {len(matches)} combinaciones coinciden con {target}")
    return matches[0]


def main() -> None:
    X_train, y_train, _, _, meta_train, _ = load_train_test()
    cv = predefined_cv(meta_train)

    per_fold: dict[str, pd.DataFrame] = {}
    params_used: dict[str, dict] = {}
    for name, (_, pipe, space) in search_spaces().items():
        params = best_params(name, space)
        params_used[name] = params
        pipe.set_params(memory=None, **params)
        print(f"Validación cruzada: {name}...", flush=True)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring(), n_jobs=-1)
        per_fold[name] = pd.DataFrame({m: scores[f"test_{m}"] for m in METRICS})

    lr_name, lgbm_name = list(per_fold)
    lr, lgbm = per_fold[lr_name], per_fold[lgbm_name]

    lines = [
        "# Comparación pareada: regresión logística vs. LightGBM (ajustados)",
        "",
        "Mismos 5 folds agrupados por canal (`folds_v1.csv`), solo train. "
        "Hiperparámetros: los mejores de `RandomizedSearchCV` (ver `tuning.md`).",
        "",
    ]
    for name, params in params_used.items():
        shown = ", ".join(f"`{k.removeprefix('model__')}={v:.4g}`" for k, v in sorted(params.items()))
        lines.append(f"- {name}: {shown}")
    lines += [
        "",
        "## F1 macro por fold",
        "",
        "| Fold | Regresión logística | LightGBM | Diferencia (RL - LGBM) |",
        "|---:|---:|---:|---:|",
    ]
    diff = lr["f1_macro"] - lgbm["f1_macro"]
    for i in range(len(lr)):
        lines.append(
            f"| {i} | {lr['f1_macro'][i]:.3f} | {lgbm['f1_macro'][i]:.3f} | {diff[i]:+.3f} |"
        )

    lines += [
        "",
        "## Resumen pareado",
        "",
        "| Métrica | RL (media ± d.e.) | LightGBM (media ± d.e.) | Dif. media ± d.e. | Folds a favor de RL | t pareada | p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    summary = {}
    for m in METRICS:
        d = lr[m] - lgbm[m]
        t, p = ttest_rel(lr[m], lgbm[m])
        summary[m] = (d.mean(), d.std(ddof=1), int((d > 0).sum()), t, p)
        lines.append(
            f"| {m} | {lr[m].mean():.3f} ± {lr[m].std(ddof=0):.3f} | "
            f"{lgbm[m].mean():.3f} ± {lgbm[m].std(ddof=0):.3f} | "
            f"{d.mean():+.3f} ± {d.std(ddof=1):.3f} | {summary[m][2]}/{len(d)} | {t:.2f} | {p:.3f} |"
        )

    f1_mean, f1_sd, f1_wins, _, f1_p = summary["f1_macro"]
    rr_mean, _, rr_wins, _, rr_p = summary["recall_restricted"]
    lines += [
        "",
        "## Lectura",
        "",
        f"- F1 macro: la regresión logística supera a LightGBM en {f1_wins} de 5 folds, "
        f"con una diferencia media de {f1_mean:+.3f} (p = {f1_p:.3f}).",
        f"- Recall de restricted: diferencia media de {rr_mean:+.3f}, a favor de la regresión "
        f"logística en {rr_wins} de 5 folds (p = {rr_p:.3f}).",
        "- Con 5 folds la prueba t tiene poca potencia y los folds no son del todo "
        "independientes (comparten entrenamiento), así que el p-valor es orientativo. "
        "La elección de la regresión logística se apoya además en su interpretabilidad "
        "y en que usa el TF-IDF completo.",
        "",
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
