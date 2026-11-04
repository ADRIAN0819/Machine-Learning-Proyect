"""Representación de texto y metadatos para los modelos de la Etapa 2.

Construye ``ColumnTransformer`` de scikit-learn sobre las columnas de
``data_processing/dataset_features_v1.csv``. Todas las transformaciones que
aprenden de los datos (vocabulario e IDF de TF-IDF, escalado, SVD) se ajustan
dentro del ``Pipeline``; con ``predefined_cv`` se ajustan solo en los folds de
entrenamiento, sin fuga hacia validación ni prueba.

Tres vistas, para el ablation study (texto vs. metadatos vs. combinado):

* ``"text"``: TF-IDF de título (1-2 gramas), descripción (1-2 gramas) y tags
  (cada tag completo es un token, p. ej. ``"for kids"``, como en el EDA).
* ``"metadata"``: duración, conteos en log(1+x), razones por vista, longitud
  del título y one-hot de categoría, definición, subtítulos y licencia.
* ``"combined"``: ambas. Con ``svd_components`` el texto se reduce con
  ``TruncatedSVD`` (útil para LightGBM / Random Forest).

Ejemplo::

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from modeling.features import build_preprocessor

    model = make_pipeline(build_preprocessor("text"),
                          LogisticRegression(max_iter=2000, class_weight="balanced"))

Las funciones auxiliares están a nivel de módulo para que los pipelines se
puedan guardar con ``joblib``.

Ejecutar ``python -m modeling.features`` genera ``reports/text_features.md``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TEXT_FEATURES = ("title", "description", "tags")
CATEGORICAL_FEATURES = ("category_id", "definition", "caption", "licensed_content")
NUMERIC_FEATURES = (
    "title_chars",
    "title_words",
    "duration_seconds",
    "view_count_log1p",
    "like_count_log1p",
    "dislike_count_log1p",
    "comment_count_log1p",
    "likes_per_view",
    "comments_per_view",
)
View = Literal["text", "metadata", "combined"]


def split_tags(value: str) -> list[str]:
    """Tokenizador de tags: separa por ``|`` y conserva cada tag completo."""
    return [tag for tag in value.split("|") if tag]


def _log_duration(values):
    return np.log1p(np.asarray(values, dtype=float))


def text_vectorizers(
    title_max_features: int = 20_000,
    description_max_features: int = 30_000,
    tags_max_features: int = 20_000,
) -> list[tuple[str, TfidfVectorizer, str]]:
    common = dict(sublinear_tf=True, strip_accents="unicode", dtype=np.float32)
    return [
        (
            "title_tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_features=title_max_features, **common
            ),
            "title",
        ),
        (
            "description_tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.9,
                max_features=description_max_features,
                **common,
            ),
            "description",
        ),
        (
            "tags_tfidf",
            TfidfVectorizer(
                tokenizer=split_tags,
                token_pattern=None,
                lowercase=False,  # ya normalizados a minúsculas en preprocessing
                min_df=2,
                max_features=tags_max_features,
                **common,
            ),
            "tags",
        ),
    ]


def metadata_transformers() -> list[tuple[str, object, list[str]]]:
    numeric = list(NUMERIC_FEATURES)
    return [
        ("numeric", StandardScaler(), numeric),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", min_frequency=5),
            list(CATEGORICAL_FEATURES),
        ),
    ]


def build_preprocessor(
    view: View = "combined",
    svd_components: int | None = None,
    random_state: int = 42,
) -> ColumnTransformer | Pipeline:
    """Devuelve el transformador de features para la vista indicada.

    ``svd_components`` reduce el bloque de texto (solo ``"text"`` o
    ``"combined"``) a una matriz densa de esa dimensión.
    """
    if view not in ("text", "metadata", "combined"):
        raise ValueError(f"Vista desconocida: {view}")

    text_block = ColumnTransformer(text_vectorizers(), sparse_threshold=1.0)
    if svd_components:
        text_block = Pipeline(
            [
                ("tfidf", text_block),
                ("svd", TruncatedSVD(n_components=svd_components, random_state=random_state)),
            ]
        )

    if view == "text":
        return text_block
    if view == "metadata":
        return ColumnTransformer(metadata_transformers())
    return ColumnTransformer(
        [("text", text_block, list(TEXT_FEATURES)), *metadata_transformers()],
        sparse_threshold=0.0 if svd_components else 0.3,
    )


def top_terms_by_class(
    X: pd.DataFrame, y: pd.Series, column: str, top_k: int = 8
) -> pd.DataFrame:
    """Términos con mayor chi² de cada clase (uno contra el resto), solo en train."""
    name, vectorizer, _ = next(v for v in text_vectorizers() if v[2] == column)
    matrix = vectorizer.fit_transform(X[column])
    vocabulary = np.array(vectorizer.get_feature_names_out())
    rows = {}
    for label in sorted(y.unique()):
        target = (y == label).to_numpy()
        scores, _ = chi2(matrix, target)
        scores = np.nan_to_num(scores)
        mean_in = np.asarray(matrix[target].mean(axis=0)).ravel()
        mean_out = np.asarray(matrix[~target].mean(axis=0)).ravel()
        scores[mean_in <= mean_out] = 0  # solo términos asociados positivamente
        best = np.argsort(scores)[::-1][:top_k]
        rows[label] = ", ".join(vocabulary[best])
    return pd.DataFrame.from_dict(rows, orient="index", columns=[column])


def main() -> None:
    from modeling.splits import MODELING_DIR, load_train_test

    X_train, y_train, X_test, _, _, _ = load_train_test()
    lines = [
        "# Representación de texto y metadatos (v1)",
        "",
        "Dimensiones ajustando las transformaciones con todo train "
        f"({len(X_train):,} videos); en validación cruzada se ajustan por fold.",
        "",
        "| Bloque | Columnas generadas |",
        "|---|---:|",
    ]
    for name, vectorizer, column in text_vectorizers():
        n = vectorizer.fit(X_train[column]).get_feature_names_out().size
        lines.append(f"| TF-IDF {column} | {n:,} |")
    meta = build_preprocessor("metadata").fit(X_train)
    lines.append(f"| Metadatos (escalados + one-hot) | {len(meta.get_feature_names_out()):,} |")
    combined = build_preprocessor("combined").fit(X_train)
    shape = combined.transform(X_test).shape
    lines.append(f"| **Combinado** | **{shape[1]:,}** |")
    lines += [
        "",
        "Términos más asociados a cada clase en train (chi², uno contra el resto):",
        "",
    ]
    for column in ("tags", "title"):
        table = top_terms_by_class(X_train, y_train, column)
        lines += [f"| Clase | {column} |", "|---|---|"]
        lines += [f"| {label} | {terms} |" for label, terms in table[column].items()]
        lines.append("")

    report = "\n".join(lines)
    path = MODELING_DIR / "reports" / "text_features.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
