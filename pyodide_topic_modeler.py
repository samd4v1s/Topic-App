"""Topic modeling for Pyodide using scikit-learn only.

This module implements a topic modeling pipeline that runs entirely in the browser
using Pyodide. Documents are embedded via LSA (Latent Semantic Analysis) using
TF-IDF + TruncatedSVD, then clustered and analyzed to extract topic words.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class PyodideTopicModeler:
    """A topic modeler for Pyodide using scikit-learn only.

    This class implements a simplified topic modeling pipeline optimized for
    in-browser execution. Embeddings are computed client-side via JavaScript
    and passed to this class for dimensionality reduction, clustering, and
    topic word extraction using TF-IDF.
    """

    def __init__(
        self,
        n_components: int = 5,
        clustering_method: str = "dbscan",
        eps: float = 0.5,
        min_samples: int = 2,
        n_clusters: int = 5,
        max_features: int = 1000,
    ) -> None:
        """Initialize the topic modeling pipeline components.

        Args:
            n_components: Number of dimensions for PCA reduction.
            clustering_method: Either "dbscan" or "kmeans".
            eps: DBSCAN epsilon parameter (neighborhood radius).
            min_samples: DBSCAN minimum samples to form a core point.
            n_clusters: Number of clusters for KMeans (if used).
            max_features: Maximum features for TfidfVectorizer.

        Raises:
            ValueError: If clustering_method is not "dbscan" or "kmeans".
        """
        self.n_components = n_components
        self.clustering_method = clustering_method
        self.eps = eps
        self.min_samples = min_samples
        self.n_clusters = n_clusters
        self.max_features = max_features

        self.pca_model = PCA(n_components=n_components)

        if clustering_method == "dbscan":
            self.clustering_model: Any = DBSCAN(eps=eps, min_samples=min_samples)
        elif clustering_method == "kmeans":
            self.clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
        else:
            raise ValueError(f"Unknown clustering method: {clustering_method}")

        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=max_features
        )

        self.cluster_labels_: np.ndarray | None = None
        self.topic_words_: dict[int, list[str]] | None = None
        self.documents_: list[str] | None = None
        self._is_fitted = False

    def fit_transform(self, docs: list[str]) -> tuple[np.ndarray, dict[int, list[str]]]:
        """Fit the topic model on documents using LSA embeddings.

        This method generates embeddings via TF-IDF + TruncatedSVD (LSA),
        then performs dimensionality reduction via PCA, clustering via DBSCAN
        or KMeans, and topic word extraction via TF-IDF.

        Args:
            docs: List of text documents.

        Returns:
            A tuple of (cluster_labels, topic_words_dict) where topic_words_dict
            maps cluster ID to top 5 TF-IDF words for that cluster.

        Raises:
            ValueError: If fewer than 2 documents are provided.
        """
        if len(docs) < 2:
            raise ValueError("At least 2 documents are required for topic modeling.")

        self.documents_ = docs

        embeddings_array = self._generate_lsa_embeddings(docs)

        reduced_embeddings = self.pca_model.fit_transform(embeddings_array)

        self.cluster_labels_ = self.clustering_model.fit_predict(reduced_embeddings)

        self.topic_words_ = self._extract_topic_words()
        self._is_fitted = True

        return self.cluster_labels_, self.topic_words_

    def _generate_lsa_embeddings(self, docs: list[str]) -> np.ndarray:
        """Generate LSA embeddings from raw documents using TF-IDF + TruncatedSVD.

        LSA (Latent Semantic Analysis) provides semantically meaningful embeddings
        by decomposing the TF-IDF document-term matrix via truncated SVD.

        Args:
            docs: List of text documents.

        Returns:
            Dense numpy array of shape (len(docs), n_components) containing embeddings.
        """
        tfidf_vectorizer = TfidfVectorizer(
            stop_words="english", max_features=2000, min_df=1, max_df=0.95
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(docs)

        n_components = min(50, len(docs) - 1)
        lsa = TruncatedSVD(n_components=n_components, random_state=42)
        embeddings = lsa.fit_transform(tfidf_matrix)

        return embeddings.astype(np.float32)

    def _extract_topic_words(self) -> dict[int, list[str]]:
        """Extract top words for each topic cluster using TF-IDF.

        For each cluster, documents are concatenated and TF-IDF is computed
        to identify the 5 highest-scoring words in the cluster.

        Returns:
            A dictionary mapping cluster ID to list of top 5 words.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self.cluster_labels_ is None or self.documents_ is None:
            raise RuntimeError("Model must be fitted before extracting topic words.")

        topic_words: dict[int, list[str]] = {}
        unique_labels = set(self.cluster_labels_)

        for cluster_id in unique_labels:
            if cluster_id == -1:
                continue

            cluster_indices = np.where(self.cluster_labels_ == cluster_id)[0]
            cluster_docs = [self.documents_[i] for i in cluster_indices]

            if not cluster_docs:
                continue

            cluster_text = " ".join(cluster_docs)

            try:
                tfidf_matrix = self.vectorizer.fit_transform([cluster_text])
                feature_names = self.vectorizer.get_feature_names_out()
                scores = tfidf_matrix.toarray()[0]

                top_indices = np.argsort(scores)[-5:][::-1]
                top_words = [
                    feature_names[i] for i in top_indices if scores[i] > 0
                ]

                topic_words[int(cluster_id)] = top_words
            except Exception:
                topic_words[int(cluster_id)] = []

        return topic_words

    def get_topic_info(self) -> pd.DataFrame:
        """Return topic information as a pandas DataFrame.

        Returns:
            A DataFrame with columns: Topic, Size, Words. Topic is the cluster ID,
            Size is the number of documents in that cluster, and Words is a
            comma-separated string of the top words.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self._is_fitted or self.cluster_labels_ is None:
            raise RuntimeError(
                "The topic model must be fitted before retrieving topic information."
            )

        topic_data: list[dict[str, Any]] = []

        for topic_id in sorted(set(self.cluster_labels_)):
            if topic_id == -1:
                continue

            size = int(np.sum(self.cluster_labels_ == topic_id))
            words = self.topic_words_.get(topic_id, [])
            words_str = ", ".join(words) if words else "N/A"

            topic_data.append({"Topic": topic_id, "Size": size, "Words": words_str})

        return pd.DataFrame(topic_data)

    def export_model_state(self) -> str:
        """Export the fitted model state as a JSON string for browser storage.

        The state includes cluster labels, topic words, and original documents.
        This JSON can be saved to localStorage or IndexedDB in the browser and
        later restored with import_model_state().

        Returns:
            JSON string containing cluster labels, topic words, and documents.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("The topic model must be fitted before exporting.")

        state = {
            "cluster_labels": self.cluster_labels_.tolist(),
            "topic_words": {str(k): v for k, v in self.topic_words_.items()},
            "documents": self.documents_,
        }

        return json.dumps(state)

    def import_model_state(self, json_str: str) -> None:
        """Import a previously exported model state from JSON.

        This allows restoring a saved model that was previously exported
        via export_model_state(). The model will be marked as fitted and
        ready to use.

        Args:
            json_str: JSON string containing model state.

        Raises:
            ValueError: If the JSON is malformed or missing required fields.
        """
        try:
            state = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}")

        required_fields = ["cluster_labels", "topic_words", "documents"]
        if not all(k in state for k in required_fields):
            raise ValueError(
                f"Missing required fields in model state JSON. Expected: {required_fields}"
            )

        self.cluster_labels_ = np.array(state["cluster_labels"], dtype=np.int64)
        self.topic_words_ = {int(k): v for k, v in state["topic_words"].items()}
        self.documents_ = state["documents"]
        self._is_fitted = True
