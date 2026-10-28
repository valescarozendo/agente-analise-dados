import json
import numpy as np
import pandas as pd
from analista_dados.profiler import ProfileConfig, build_profile


def make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "regiao": ["N", "N", "S", "S", "S", None, "N", "S", "N", "S"],
            "valor": [10.0, 12.0, 11.0, 13.0, 12.0, 11.0, 10.0, 12.0, 11.0, 500.0],
            "qtd": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "custo": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, np.nan],
            "data": pd.date_range("2024-01-01", periods=10, freq="D"),
        }
    )


def test_overview_counts_and_column_types():
    overview = build_profile(make_df())["overview"]
    assert overview["rows"] == 10
    assert overview["columns"] == 5
    assert overview["numeric_columns"] == ["valor", "qtd", "custo"]


def test_outliers_are_detected_by_iqr():
    assert build_profile(make_df())["numeric"]["valor"]["outliers_iqr"] == 1


def test_profile_is_strict_json_serializable():
    json.dumps(build_profile(make_df()), allow_nan=False)