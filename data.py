import pandas as pd
import logging


def load_survey_data(file_path):
    """Load a CSV file and normalise column names to lowercase."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower()
    return df


def select_columns(df, keyword):
    """Return the subset of columns whose names contain the given keyword."""
    cols = [c for c in df.columns if keyword in c]
    return df[cols]


def filter_spurious_responses(df):
    """Remove rows where the total of maximum columns is less than the
    total of minimum columns."""
    min_sums = select_columns(df, 'minimum').sum(axis=1)
    max_sums = select_columns(df, 'maximum').sum(axis=1)

    spurious_mask = max_sums < min_sums
    num_dropped = int(spurious_mask.sum())

    if num_dropped > 0:
        logging.warning(f"Dropped {num_dropped} spurious responses (Min > Max in at least one phase).")
    else:
        logging.info("No spurious responses detected.")

    return df[~spurious_mask].reset_index(drop=True)
