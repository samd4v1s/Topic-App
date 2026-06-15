"""Advanced topic modeling utilities built on BERTopic."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, OpenAI
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


class AdvancedTopicModeler:
    """A BERTopic-based topic modeler with embedding, reduction, and clustering.

    The class follows the advanced topic modeling pipeline described in the
    Towards Data Science article: embedding, dimensionality reduction,
    clustering, vectorization, weighting, and topic representation generation.
    """

    def __init__(
        self,
        embedding_model_name: str = "thenlper/gte-small",
        use_openai_representation: bool = False,
        openai_api_key: str | None = None,
        openai_model: str = "gpt-4o-mini",
    ) -> None:
        """Initialize the pipeline components used by BERTopic.

        Args:
            embedding_model_name: Hugging Face model identifier for the
                sentence embedding model.
            use_openai_representation: Whether to include an OpenAI-based topic
                label generator in the representation model chain.
            openai_api_key: Optional API key for the OpenAI representation model.
            openai_model: Name of the OpenAI chat model to use for topic labels.
        """
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.umap_model = UMAP(n_neighbors=5, metric="cosine")
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=10,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        self.vectorizer_model = CountVectorizer(stop_words="english")
        self.ctfidf_model = ClassTfidfTransformer()

        self.representation_models: list[Any] = [KeyBERTInspired()]
        if use_openai_representation:
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY") or ""
            self.representation_models.append(
                OpenAI(model=openai_model, chat=True, api_key=api_key)
            )

        self.model: BERTopic | None = None
        self._is_fitted = False

    def fit_transform(self, docs: list[str]) -> tuple[list[int], list[float]]:
        """Fit the BERTopic pipeline on a list of documents.

        Args:
            docs: The documents to cluster and describe.

        Returns:
            A tuple containing the topic assignments and document probabilities.

        Raises:
            ValueError: If no documents are provided.
        """
        if not docs:
            raise ValueError("At least one document is required for fitting.")

        self.model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            representation_model=self.representation_models,
            verbose=False,
        )
        topics, probabilities = self.model.fit_transform(docs)
        self._is_fitted = True
        return topics, probabilities

    def get_topic_info(self) -> pd.DataFrame:
        """Return topic information as a pandas DataFrame.

        Returns:
            The topic information table produced by BERTopic.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self.model is None:
            raise RuntimeError("The topic model must be fitted before retrieving topic information.")

        return self.model.get_topic_info()

    def save_model(self, model_name: str, base_dir: str = "./saved_models") -> Path:
        """Persist a fitted BERTopic model to disk with safetensors serialization.

        Args:
            model_name: The local name used for the saved model directory.
            base_dir: The directory where the model should be stored.

        Returns:
            The path to the saved model directory.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self.model is None:
            raise RuntimeError("The topic model must be fitted before saving.")

        save_path = Path(base_dir) / model_name
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path), serialization="safetensors", save_ctfidf=True)
        return save_path

    @classmethod
    def load_model(cls, model_name: str, base_dir: str = "./saved_models") -> "AdvancedTopicModeler":
        """Load a saved BERTopic model from disk.

        Args:
            model_name: The local name of the saved model directory.
            base_dir: The parent directory containing the saved model.

        Returns:
            A new instance of AdvancedTopicModeler with the loaded model.

        Raises:
            FileNotFoundError: If the requested model cannot be found.
        """
        model_path = Path(base_dir) / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        loaded_model = BERTopic.load(str(model_path))
        instance = cls(use_openai_representation=False)
        instance.model = loaded_model
        instance._is_fitted = True
        return instance
