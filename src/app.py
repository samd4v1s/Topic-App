"""Streamlit web app for training and saving topic models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.topic_modeler import AdvancedTopicModeler


st.set_page_config(page_title="Team Topic Modeler", layout="wide")

if "topic_model" not in st.session_state:
    st.session_state.topic_model: AdvancedTopicModeler | None = None
if "is_model_trained" not in st.session_state:
    st.session_state.is_model_trained = False


def _get_saved_models(base_dir: str = "./saved_models") -> list[str]:
    """List available saved model directories."""
    models_dir = Path(base_dir)
    if not models_dir.exists():
        return []
    return sorted([path.name for path in models_dir.iterdir() if path.is_dir()])


def _get_text_column_candidates(frame: pd.DataFrame) -> list[str]:
    """Return columns that are likely to contain free-form text."""
    candidate_columns: list[str] = []
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue

        non_null_values = series.dropna()
        if non_null_values.empty:
            continue

        string_values = non_null_values.astype("string").str.strip()
        non_empty_count = int((string_values != "").sum())
        if non_empty_count == 0:
            continue

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
            or non_empty_count / len(non_null_values) >= 0.5
        ):
            candidate_columns.append(column)

    return candidate_columns


st.title("Team Topic Modeler")
st.caption("Upload a CSV, select a text column, and train a BERTopic model in seconds.")

with st.sidebar:
    st.header("Saved Models")
    saved_models = _get_saved_models()
    if saved_models:
        selected_model = st.selectbox("Load a previously saved model", saved_models)
        if st.button("Load Model"):
            try:
                loaded_model = AdvancedTopicModeler.load_model(selected_model)
                st.session_state.topic_model = loaded_model
                st.session_state.is_model_trained = True
                st.success(f"Loaded model: {selected_model}")
            except Exception as exc:  # pragma: no cover - UI error handling
                st.error(f"Could not load model: {exc}")
    else:
        st.warning("No saved models found in ./saved_models.")

st.subheader("Upload Documents")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data_frame = pd.read_csv(uploaded_file)
    except Exception as exc:  # pragma: no cover - UI error handling
        st.error(f"Unable to read the uploaded CSV: {exc}")
        st.stop()

    st.dataframe(data_frame.head(), use_container_width=True)

    if data_frame.empty:
        st.warning("The uploaded CSV is empty.")
        st.stop()

    text_columns = _get_text_column_candidates(data_frame)
    if not text_columns:
        st.warning(
            "No text-like columns were found in the uploaded file. Try a column with free-form text values."
        )
        st.stop()

    selected_column = st.selectbox("Select the text column to analyze", text_columns)

    if st.button("Analyze Documents"):
        if not selected_column:
            st.error("Please select a text column before analyzing documents.")
        else:
            documents = [str(value).strip() for value in data_frame[selected_column].tolist() if str(value).strip()]
            if not documents:
                st.warning("The selected column contains no usable text values.")
            else:
                with st.spinner("Training topic model..."):
                    model = AdvancedTopicModeler()
                    model.fit_transform(documents)
                    st.session_state.topic_model = model
                    st.session_state.is_model_trained = True

                st.success("Topic model trained successfully.")
                topic_info = st.session_state.topic_model.get_topic_info()
                st.subheader("Topic Results")
                st.dataframe(topic_info, use_container_width=True)

if st.session_state.get("topic_model") is not None:
    st.divider()
    st.subheader("Save Model")
    model_name = st.text_input("Model name", placeholder="e.g. product_feedback_model")
    if st.button("Save Model to Memory"):
        if not model_name.strip():
            st.warning("Please provide a model name before saving.")
        else:
            try:
                save_path = st.session_state.topic_model.save_model(model_name.strip())
                st.success(f"Model saved successfully to {save_path}")
            except Exception as exc:  # pragma: no cover - UI error handling
                st.error(f"Could not save model: {exc}")
