"""Self-check for the Pyodide topic logic (runs locally with plain scikit-learn).

Feeds synthetic embeddings, so it exercises the reduce/cluster/label pipeline
without needing transformers.js or a browser.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs"))

from topic_pyodide import fit_topics_from_embeddings, text_columns  # noqa: E402

CSV = "id,note\n" + "\n".join(
    f"{i}," + ("cats and dogs are great pets" if i % 2 else "stock market shares and bonds")
    for i in range(20)
)


def test_text_columns_skips_numeric():
    assert text_columns(CSV) == ["note"]


def test_clusters_two_groups():
    docs = ["cats dogs pets animals"] * 10 + ["stocks bonds market shares"] * 10
    rng = np.random.RandomState(0)
    blob_a = rng.normal(0, 0.01, (10, 8)) + np.eye(8)[0] * 5
    blob_b = rng.normal(0, 0.01, (10, 8)) + np.eye(8)[1] * 5
    embeddings = np.vstack([blob_a, blob_b]).tolist()

    topics = fit_topics_from_embeddings(embeddings, docs, min_cluster_size=3)
    real = [t for t in topics if not t["outlier"]]
    assert len(real) == 2
    assert sum(t["count"] for t in topics) == 20
    words = " ".join(t["words"] for t in real)
    assert "pets" in words and "market" in words
