"""Pyodide-side topic modeling: reduce + cluster + label, pure scikit-learn.

The transformer embeddings are produced in JavaScript by transformers.js (ONNX,
since PyTorch can't run in WASM) and handed to this module. Everything after the
embedding step mirrors BERTopic: TruncatedSVD for dimensionality reduction,
sklearn's HDBSCAN for clustering, and a class-based TF-IDF to label each topic.

Importable for local testing — feed it synthetic embeddings, no browser needed.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def text_columns(csv_text: str) -> list[str]:
    """Return non-numeric columns that contain some free-form text."""
    frame = pd.read_csv(io.StringIO(csv_text))
    cols = []
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        if series.dropna().astype(str).str.strip().ne("").any():
            cols.append(str(column))
    return cols


def extract_docs(csv_text: str, column: str) -> list[str]:
    """Pull the cleaned, non-empty text values from one column."""
    frame = pd.read_csv(io.StringIO(csv_text))
    return [
        text
        for value in frame[column].tolist()
        if (text := str(value).strip()) and text.lower() != "nan"
    ]


def fit_topics_from_embeddings(
    embeddings, docs: list[str], min_cluster_size: int = 5, n_words: int = 10
) -> list[dict]:
    """Reduce, cluster, and label documents using their embeddings.

    Returns a list of {topic, count, words, outlier} dicts. Topic -1 is the
    HDBSCAN outlier group, sorted last.
    """
    docs = list(docs)
    points = np.asarray(embeddings, dtype=float)
    if points.ndim != 2 or points.shape[0] != len(docs):
        raise ValueError("Embeddings shape does not match number of documents.")
    n = points.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 documents to find topics.")

    # Mirror BERTopic's UMAP-to-5-dims step (TruncatedSVD is Pyodide-safe; UMAP isn't).
    if points.shape[1] > 5 and n > 6:
        n_comp = min(5, n - 1, points.shape[1] - 1)
        points = TruncatedSVD(n_components=n_comp, random_state=42).fit_transform(points)

    mcs = max(2, min(int(min_cluster_size), n))
    labels = HDBSCAN(min_cluster_size=mcs).fit_predict(points)
    return _ctfidf_topics(labels, docs, n_words)


def _ctfidf_topics(labels, docs: list[str], n_words: int) -> list[dict]:
    """Label each cluster with its most distinctive words via class-based TF-IDF."""
    labels = np.asarray(labels)
    uniq = sorted(set(labels.tolist()))
    joined = [
        " ".join(docs[i] for i in range(len(docs)) if labels[i] == lab) for lab in uniq
    ]

    terms = []
    matrix = None
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(joined).toarray()
        terms = vectorizer.get_feature_names_out()
    except ValueError:  # empty vocabulary (e.g. all stop words)
        pass

    topics = []
    for row, lab in enumerate(uniq):
        if matrix is not None and len(terms):
            order = matrix[row].argsort()[::-1][:n_words]
            top = [terms[i] for i in order if matrix[row][i] > 0]
        else:
            top = []
        topics.append(
            {
                "topic": int(lab),
                "count": int((labels == lab).sum()),
                "words": ", ".join(top) if top else "(no distinctive words)",
                "outlier": bool(lab == -1),
            }
        )
    topics.sort(key=lambda t: (t["outlier"], -t["count"]))
    return topics


def fit_topics_json(
    csv_text: str, column: str, embeddings, min_cluster_size: int = 5
) -> str:
    """JSON wrapper: re-extract docs from the CSV and cluster with the embeddings."""
    docs = extract_docs(csv_text, column)
    return json.dumps(fit_topics_from_embeddings(embeddings, docs, min_cluster_size))
