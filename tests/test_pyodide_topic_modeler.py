"""Tests for the Pyodide-compatible topic modeler."""

from __future__ import annotations

import json

import pytest

from src.pyodide_topic_modeler import PyodideTopicModeler


@pytest.fixture
def sample_documents() -> list[str]:
    """Provide sample documents for testing."""
    return [
        "machine learning algorithms process data",
        "neural networks solve classification problems",
        "deep learning uses multiple layers",
        "regression predicts continuous values",
        "classification assigns discrete labels",
        "clustering groups similar data points",
    ]


def test_initialization_creates_pipeline_components() -> None:
    """The model should initialize its core components."""
    modeler = PyodideTopicModeler(clustering_method="dbscan")

    assert modeler.pca_model is not None
    assert modeler.clustering_model is not None
    assert modeler.vectorizer is not None
    assert modeler.cluster_labels_ is None
    assert modeler.topic_words_ is None


def test_invalid_clustering_method_raises_error() -> None:
    """Invalid clustering method should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown clustering method"):
        PyodideTopicModeler(clustering_method="invalid_method")


def test_fit_transform_returns_labels_and_words(sample_documents: list[str]) -> None:
    """Fitting should return cluster labels and topic words using LSA embeddings."""
    modeler = PyodideTopicModeler(clustering_method="dbscan", eps=1.0, min_samples=1)
    labels, words = modeler.fit_transform(sample_documents)

    assert labels is not None
    assert isinstance(words, dict)
    assert len(labels) == len(sample_documents)
    assert modeler._is_fitted


def test_fit_transform_with_too_few_documents_raises_error() -> None:
    """Fitting with fewer than 2 documents should raise ValueError."""
    modeler = PyodideTopicModeler()
    single_doc = ["only one document here"]

    with pytest.raises(ValueError, match="At least 2 documents"):
        modeler.fit_transform(single_doc)


def test_get_topic_info_returns_dataframe(sample_documents: list[str]) -> None:
    """get_topic_info should return a valid DataFrame after fitting."""
    modeler = PyodideTopicModeler(clustering_method="kmeans", n_clusters=2)
    modeler.fit_transform(sample_documents)

    topic_info = modeler.get_topic_info()

    assert topic_info is not None
    assert "Topic" in topic_info.columns
    assert "Size" in topic_info.columns
    assert "Words" in topic_info.columns


def test_export_and_import_model_state_round_trip(sample_documents: list[str]) -> None:
    """Model state should be exportable and importable from JSON."""
    modeler1 = PyodideTopicModeler(clustering_method="kmeans", n_clusters=2)
    modeler1.fit_transform(sample_documents)

    exported_json = modeler1.export_model_state()
    assert isinstance(exported_json, str)

    parsed = json.loads(exported_json)
    assert "cluster_labels" in parsed
    assert "topic_words" in parsed
    assert "documents" in parsed

    modeler2 = PyodideTopicModeler()
    modeler2.import_model_state(exported_json)

    assert modeler2._is_fitted
    assert len(modeler2.cluster_labels_) == len(sample_documents)
    assert modeler2.documents_ == sample_documents


def test_export_unfitted_model_raises_error() -> None:
    """Exporting an unfitted model should raise RuntimeError."""
    modeler = PyodideTopicModeler()

    with pytest.raises(RuntimeError, match="must be fitted before exporting"):
        modeler.export_model_state()


def test_import_invalid_json_raises_error() -> None:
    """Importing malformed JSON should raise ValueError."""
    modeler = PyodideTopicModeler()

    with pytest.raises(ValueError, match="Invalid JSON"):
        modeler.import_model_state("not valid json")


def test_import_missing_fields_raises_error() -> None:
    """Importing JSON with missing fields should raise ValueError."""
    modeler = PyodideTopicModeler()
    incomplete_json = json.dumps({"cluster_labels": [0, 1, 0]})

    with pytest.raises(ValueError, match="Missing required fields"):
        modeler.import_model_state(incomplete_json)
