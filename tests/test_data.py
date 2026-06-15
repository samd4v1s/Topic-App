import pandas as pd
import pytest
from archive.data import load_survey_data, select_columns, filter_spurious_responses


def test_select_columns_finds_matching_columns():
    df = pd.DataFrame({'phase 1 minimum': [1], 'phase 1 maximum': [5], 'team': ['A']})
    result = select_columns(df, 'minimum')
    assert list(result.columns) == ['phase 1 minimum']


def test_select_columns_returns_empty_when_no_match():
    df = pd.DataFrame({'phase 1 minimum': [1]})
    result = select_columns(df, 'maximum')
    assert result.empty


def test_keeps_valid_rows():
    df = pd.DataFrame({
        'phase 1 minimum': [2, 4],
        'phase 1 maximum': [10, 12],
        'phase 2 minimum': [3, 5],
        'phase 2 maximum': [8, 15],
    })
    result = filter_spurious_responses(df)
    assert len(result) == 2


def test_drops_row_where_total_max_less_than_total_min():
    df = pd.DataFrame({
        'phase 1 minimum': [2, 20],
        'phase 1 maximum': [10, 1],
        'phase 2 minimum': [3, 30],
        'phase 2 maximum': [8, 2],
    })
    result = filter_spurious_responses(df)
    assert len(result) == 1
    assert result.iloc[0]['phase 1 minimum'] == 2


def test_load_survey_data_lowercases_columns(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("Phase 1 Minimum,Phase 1 Maximum\n2,10\n")
    df = load_survey_data(csv)
    assert list(df.columns) == ['phase 1 minimum', 'phase 1 maximum']


def test_load_survey_data_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_survey_data(tmp_path / "does_not_exist.csv")
