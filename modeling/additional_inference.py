"""Comprobación de consistencia en los conjuntos adicionales (PI2 y PI3).

Aplica un modelo ya entrenado sobre ``dataset_consolidado_parts/*.csv.gz``,
excluyendo todo video del ground truth, y compara sus predicciones con la
columna ``prediction`` del clasificador original de Papadamou et al.

* Nuestro modelo predice 4 clases; se considera *inapropiado* si predice
  ``disturbing`` o ``restricted`` (mismo criterio binario que ``prediction``).
* Fuera del ground truth no hay etiquetas humanas: los resultados miden
  **consistencia**, no recall ni tasa de falsos positivos.
* En el consolidado cada video aparece una sola vez, con un único
  ``source_dataset``; un video presente en varios conjuntos se cuenta en uno.

Uso::

    # Con el modelo final (pipeline de sklearn guardado con joblib que acepta
    # el DataFrame de features de dataset_features_v1.csv):
    python -m modeling.additional_inference --model modeling/models/final.joblib

    # Prueba de extremo a extremo con un modelo de referencia (TF-IDF + RL):
    python -m modeling.additional_inference --demo --limit-per-part 20000

Salidas: ``modeling/reports/additional_consistency.md`` (resumen) y
``modeling/outputs/additional_predictions.csv.gz`` (predicción por video;
no se versiona por tamaño).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import cohen_kappa_score

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "data_processing"))

from preprocessing import MODEL_FEATURE_COLUMNS, transform_unlabeled  # noqa: E402

from modeling.splits import MODELING_DIR, load_train_test  # noqa: E402

DATA_DIR = REPOSITORY_ROOT / "dataset_disturbed_youtube _for_kids"
PARTS_GLOB = "dataset_consolidado_parts/part_*.csv.gz"
GROUNDTRUTH_PATH = DATA_DIR / "dataset_groundtruth.csv"
REPORT_PATH = MODELING_DIR / "reports" / "additional_consistency.md"
PREDICTIONS_PATH = MODELING_DIR / "outputs" / "additional_predictions.csv.gz"

INAPPROPRIATE = {"disturbing", "restricted"}
SOURCE_ORDER = ("elsagate_related", "other_child_related", "random", "popular")
RAW_COLUMNS = [
    "video_id", "title", "description", "channel_id", "category_id", "tags",
    "duration_seconds", "definition", "caption", "licensed_content",
    "view_count", "like_count", "dislike_count", "comment_count",
    "prediction", "source_dataset",
]


def iter_unlabeled(limit_per_part: int | None = None):
    """Recorre el consolidado parte por parte, sin videos del ground truth."""
    gt_ids = set(pd.read_csv(GROUNDTRUTH_PATH, usecols=["video_id"], dtype=str)["video_id"])
    for part in sorted(DATA_DIR.glob(PARTS_GLOB)):
        chunk = pd.read_csv(
            part, usecols=RAW_COLUMNS, dtype=str, keep_default_na=False,
            nrows=limit_per_part,
        )
        keep = chunk["source_dataset"].ne("groundtruth") & ~chunk["video_id"].isin(gt_ids)
        yield part.name, chunk.loc[keep].reset_index(drop=True)


def predict_unlabeled(model, limit_per_part: int | None = None) -> pd.DataFrame:
    results = []
    for name, raw in iter_unlabeled(limit_per_part):
        if raw.empty:
            continue
        X = transform_unlabeled(raw)
        predicted = model.predict(X.loc[:, list(MODEL_FEATURE_COLUMNS)])
        results.append(
            pd.DataFrame(
                {
                    "video_id": raw["video_id"],
                    "source_dataset": raw["source_dataset"],
                    "model_label": predicted,
                    "original_prediction": raw["prediction"].str.strip().str.lower(),
                }
            )
        )
        print(f"{name}: {len(raw):,} videos procesados", flush=True)
    out = pd.concat(results, ignore_index=True)
    out["model_inappropriate"] = out["model_label"].isin(INAPPROPRIATE)
    out["original_inappropriate"] = out["original_prediction"].eq("inappropriate")
    return out


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in [s for s in SOURCE_ORDER if s in set(predictions["source_dataset"])]:
        d = predictions.loc[predictions["source_dataset"].eq(source)]
        m, o = d["model_inappropriate"], d["original_inappropriate"]
        rows.append(
            {
                "Conjunto": source,
                "Videos": len(d),
                "% inapropiado (modelo)": 100 * m.mean(),
                "% inapropiado (original)": 100 * o.mean(),
                "Acuerdo %": 100 * (m == o).mean(),
                "Kappa": cohen_kappa_score(o, m) if m.nunique() > 1 or o.nunique() > 1 else float("nan"),
                "Ambos inapropiado": int((m & o).sum()),
                "Solo modelo": int((m & ~o).sum()),
                "Solo original": int((~m & o).sum()),
                **{
                    f"% {c}": 100 * d["model_label"].eq(c).mean()
                    for c in ("disturbing", "restricted", "suitable", "irrelevant")
                },
            }
        )
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, model_name: str, limited: bool) -> str:
    lines = [
        "# Consistencia en conjuntos adicionales (PI2 y PI3)",
        "",
        f"Modelo: `{model_name}`. Se excluyen todos los videos del ground truth. "
        "*Inapropiado* = `disturbing` o `restricted`. Sin etiquetas humanas: "
        "los valores miden consistencia con el clasificador original, no eficacia.",
    ]
    if limited:
        lines += ["", "> **Ejecución parcial** (`--limit-per-part`): no usar estas cifras en el informe."]
    lines += [
        "",
        "| Conjunto | Videos | % inapropiado (modelo) | % inapropiado (original) | Acuerdo % | Kappa | Ambos | Solo modelo | Solo original |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['Conjunto']} | {r['Videos']:,} | {r['% inapropiado (modelo)']:.2f} | "
            f"{r['% inapropiado (original)']:.2f} | {r['Acuerdo %']:.2f} | {r['Kappa']:.3f} | "
            f"{r['Ambos inapropiado']:,} | {r['Solo modelo']:,} | {r['Solo original']:,} |"
        )
    lines += [
        "",
        "Distribución de las 4 clases predichas por el modelo (%):",
        "",
        "| Conjunto | disturbing | restricted | suitable | irrelevant |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['Conjunto']} | {r['% disturbing']:.2f} | {r['% restricted']:.2f} | "
            f"{r['% suitable']:.2f} | {r['% irrelevant']:.2f} |"
        )
    lines += [
        "",
        "Las columnas *Solo modelo* y *Solo original* son los desacuerdos que se "
        "revisarán de forma cualitativa (H2).",
    ]
    return "\n".join(lines) + "\n"


def demo_model():
    """Modelo de referencia (no es el modelo final) para probar el flujo."""
    from modeling.leakage_check import build_model

    X_train, y_train, *_ = load_train_test()
    return build_model().fit(X_train, y_train)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=Path, help="Pipeline entrenado (.joblib)")
    group.add_argument("--demo", action="store_true", help="Usar el modelo de referencia TF-IDF + RL")
    parser.add_argument("--limit-per-part", type=int, default=None, help="Leer solo N filas por parte (prueba rápida)")
    args = parser.parse_args()

    model = demo_model() if args.demo else joblib.load(args.model)
    model_name = "demo TF-IDF + regresión logística" if args.demo else args.model.name

    predictions = predict_unlabeled(model, args.limit_per_part)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    report = write_report(summarize(predictions), model_name, args.limit_per_part is not None)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
