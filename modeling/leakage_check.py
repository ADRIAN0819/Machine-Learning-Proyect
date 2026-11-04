"""Evidencia empírica para la partición agrupada por canal.

Compara un mismo modelo simple (TF-IDF sobre título + tags + descripción y
regresión logística balanceada) evaluado con:

* ``StratifiedKFold`` aleatorio: videos del mismo canal pueden quedar en
  entrenamiento y validación a la vez.
* Los folds agrupados por canal de ``folds_v1.csv``.

Si el puntaje aleatorio es mayor, la diferencia estima cuánto se inflaría el
desempeño por memorizar rasgos del canal (fuga de datos, Sección II-J del
informe P1). Solo usa train; el conjunto de prueba no se toca.

Ejecutar: ``python -m modeling.leakage_check``
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from modeling.splits import MODELING_DIR, RANDOM_STATE, load_train_test, predefined_cv

REPORT_PATH = MODELING_DIR / "reports" / "leakage_check.md"
SCORING = {
    "f1_macro": "f1_macro",
    "recall_macro": "recall_macro",
    "accuracy": "accuracy",
}


def _join_text(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["title"] + " " + frame["tags"].str.replace("|", " ", regex=False)
        + " " + frame["description"]
    )


def build_model():
    return make_pipeline(
        FunctionTransformer(_join_text),
        TfidfVectorizer(min_df=2, max_features=50_000, ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )


def main() -> None:
    X_train, y_train, _, _, meta_train, _ = load_train_test()
    schemes = {
        "Aleatorio (StratifiedKFold)": StratifiedKFold(
            n_splits=5, shuffle=True, random_state=RANDOM_STATE
        ),
        "Agrupado por canal (folds_v1)": predefined_cv(meta_train),
    }
    rows = []
    for name, cv in schemes.items():
        scores = cross_validate(build_model(), X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        row = {"Validación": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[metric] = f"{values.mean():.3f} ± {values.std():.3f}"
            row[f"_{metric}"] = values.mean()
        rows.append(row)

    table = pd.DataFrame(rows)
    gap = table.loc[0, "_f1_macro"] - table.loc[1, "_f1_macro"]
    lines = [
        "# Fuga de datos por canal: validación aleatoria vs. agrupada",
        "",
        "Modelo: TF-IDF (1-2 gramas) sobre título + tags + descripción y regresión "
        "logística con `class_weight='balanced'`. 5 folds sobre train (3,842 videos).",
        "",
        "| Validación | F1 macro | Recall macro | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for _, r in table.iterrows():
        lines.append(f"| {r['Validación']} | {r['f1_macro']} | {r['recall_macro']} | {r['accuracy']} |")
    lines += [
        "",
        f"Diferencia en F1 macro (aleatorio − agrupado): **{gap:+.3f}**. "
        "Una diferencia positiva indica que la validación aleatoria sobreestima el "
        "desempeño porque el modelo reconoce canales vistos en entrenamiento.",
    ]
    report = "\n".join(lines) + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
