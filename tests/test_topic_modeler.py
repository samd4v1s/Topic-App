"""Tests for the advanced topic modeling module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.topic_modeler import AdvancedTopicModeler


@pytest.fixture
def sample_documents() -> list[str]:
    """Provide a small collection of example documents."""
    return [
        "Machine learning models are improving healthcare diagnostics.",
        "Natural language processing helps analyze customer feedback.",
        "Deep learning powers image recognition systems in robotics.",
        "The finance industry uses predictive analytics for fraud detection.",
        "Topic modeling is a useful technique for organizing documents.",
        "Search engines use embeddings to understand user intent.",
        "Large language models are reshaping software development workflows.",
        "Clustering algorithms group similar documents into topics.",
    ]


def test_initialization_creates_pipeline_components() -> None:
    """The model should initialize its core BERTopic components."""
    modeler = AdvancedTopicModeler(use_openai_representation=False)

    assert modeler.embedding_model is not None
    assert modeler.umap_model is not None
    assert modeler.hdbscan_model is not None
    assert modeler.vectorizer_model is not None
    assert modeler.ctfidf_model is not None
    assert modeler.representation_models is not None
    assert len(modeler.representation_models) == 1
    assert modeler.model is None


def test_fit_transform_returns_topics_and_probabilities(sample_documents: list[str]) -> None:
    """Fitting on sample text should return topics and probabilities."""
    modeler = AdvancedTopicModeler(use_openai_representation=False)
    topics, probabilities = modeler.fit_transform(sample_documents)

    assert len(topics) == len(sample_documents)
    assert len(probabilities) == len(sample_documents)
    assert modeler.model is not None

    topic_info = modeler.get_topic_info()
    assert isinstance(topic_info, pd.DataFrame)


def test_save_and_load_model_round_trip(sample_documents: list[str], tmp_path: Path) -> None:
    """The fitted model should be savable to disk and re-loadable."""
    modeler = AdvancedTopicModeler(use_openai_representation=False)
    modeler.fit_transform(sample_documents)

    saved_path = modeler.save_model("round_trip_model", base_dir=str(tmp_path))

    assert saved_path.exists()

    loaded_modeler = AdvancedTopicModeler.load_model("round_trip_model", base_dir=str(tmp_path))

    assert loaded_modeler.model is not None
    assert loaded_modeler.get_topic_info().shape[0] >= 0
