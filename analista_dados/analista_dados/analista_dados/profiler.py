"""Cálculo do perfil estatístico (matriz sintética) de um DataFrame."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProfileConfig:
    """Parâmetros do perfil estatístico."""
    top_categories: int = 5
    correlation_threshold: float = 0.3
    max_correlations: int = 10
    outlier_iqr_factor: float = 1.5
    decimals: int = 4


def build_profile(df: pd.DataFrame, config: ProfileConfig | None = None) -> dict[str, Any]:
    """Resume o DataFrame em estatísticas descritivas agregadas."""
    config = config or ProfileConfig()
    numeric = df.select_dtypes(include="number")
    temporal = df.select_dtypes(include=["datetime", "datetimetz"])
    categorical = df.drop(columns=[*numeric.columns, *temporal.columns])

    profile = {
        "overview": _overview(df, numeric, categorical, temporal),
        "data_quality": _data_quality(df),
        "numeric": _numeric_summary(numeric, config),
        "categorical": _categorical_summary(categorical, config),
        "temporal": _temporal_summary(temporal),
        "correlations": _correlations(numeric, config),
    }
    return _sanitize(profile, config.decimals)


def _overview(
    df: pd.DataFrame,
    numeric: pd.DataFrame,
    categorical: pd.DataFrame,
    temporal: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": list(numeric.columns),
        "categorical_columns": list(categorical.columns),
        "temporal_columns": list(temporal.columns),
    }


def _data_quality(df: pd.DataFrame) -> dict[str, Any]:
    total = len(df)
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    return {
        "missing_values": {
            column: {"count": int(count), "pct": count / total * 100}
            for column, count in missing.items()
        },
        "constant_columns": [c for c in df.columns if df[c].nunique(dropna=True) <= 1],
    }


def _numeric_summary(numeric: pd.DataFrame, config: ProfileConfig) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for column in numeric.columns:
        series = numeric[column].dropna()
        if series.empty:
            continue
        q1, median, q3 = series.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        mean, std = series.mean(), series.std()
        outliers = 0
        if iqr > 0:
            low = q1 - config.outlier_iqr_factor * iqr
            high = q3 + config.outlier_iqr_factor * iqr
            outliers = int(((series < low) | (series > high)).sum())
        summary[column] = {
            "count": int(series.size),
            "sum": series.sum(),
            "mean": mean,
            "std": std,
            "cv": std / abs(mean) if mean != 0 else None,
            "min": series.min(),
            "q1": q1,
            "median": median,
            "q3": q3,
            "max": series.max(),
            "skew": series.skew(),
            "outliers_iqr": outliers,
            "outliers_pct": outliers / series.size * 100,
        }
    return summary


def _categorical_summary(categorical: pd.DataFrame, config: ProfileConfig) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for column in categorical.columns:
        series = categorical[column].dropna()
        if series.empty:
            continue
        counts = series.value_counts()
        summary[column] = {
            "unique": int(counts.size),
            "top": [
                {"value": str(value), "count": int(count), "pct": count / series.size * 100}
                for value, count in counts.head(config.top_categories).items()
            ],
        }
    return summary


def _temporal_summary(temporal: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for column in temporal.columns:
        series = temporal[column].dropna()
        if series.empty:
            continue
        start, end = series.min(), series.max()
        summary[column] = {"start": start, "end": end, "span_days": (end - start).days}
    return summary


def _correlations(numeric: pd.DataFrame, config: ProfileConfig) -> list[dict[str, Any]]:
    if numeric.shape[1] < 2:
        return []
    matrix = numeric.corr(method="pearson")
    columns = list(matrix.columns)
    pairs = [
        {"a": a, "b": b, "pearson": matrix.loc[a, b]}
        for i, a in enumerate(columns)
        for b in columns[i + 1 :]
        if pd.notna(matrix.loc[a, b]) and abs(matrix.loc[a, b]) >= config.correlation_threshold
    ]
    pairs.sort(key=lambda pair: abs(pair["pearson"]), reverse=True)
    return pairs[: config.max_correlations]


def _sanitize(value: Any, decimals: int) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize(item, decimals) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, decimals) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return round(number, decimals) if math.isfinite(number) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value