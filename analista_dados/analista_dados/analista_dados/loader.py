"""Leitura e normalização de datasets tabulares."""
from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Callable
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_CSV_ENCODINGS = ("utf-8-sig", "latin-1")
_CSV_SEPARATORS = (",", ";", "\t", "|")
_SAMPLE_SIZE = 200
_PARSE_THRESHOLD = 0.95

# Números no padrão brasileiro: 1.234,56 | 1234,56 | 12
_LOCALIZED_NUMBER = re.compile(r"^-?(\d{1,3}(\.\d{3})+|\d+)(,\d+)?$")


class DatasetError(Exception):
    """Falha na leitura ou na validação do dataset."""


def _read_delimited(path: Path) -> pd.DataFrame:
    """Lê CSV/TSV detectando o separador e tentando codificações comuns."""
    for encoding in _CSV_ENCODINGS:
        try:
            with path.open(encoding=encoding) as handle:
                header = handle.readline()
            separator = max(_CSV_SEPARATORS, key=header.count)
            return pd.read_csv(path, sep=separator, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise DatasetError(f"Não foi possível decodificar '{path.name}'.")


_READERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    ".csv": _read_delimited,
    ".tsv": _read_delimited,
    ".txt": _read_delimited,
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
    ".parquet": pd.read_parquet,
    ".json": pd.read_json,
}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Carrega um dataset e normaliza nomes de colunas e tipos."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Arquivo não encontrado: {path}")

    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        supported = ", ".join(sorted(_READERS))
        raise DatasetError(f"Formato '{path.suffix}' não suportado. Use: {supported}.")

    try:
        df = reader(path)
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(f"Falha ao ler '{path.name}': {exc}") from exc

    if df.empty:
        raise DatasetError(f"O dataset '{path.name}' não contém linhas.")

    df.columns = [str(column).strip() for column in df.columns]
    df = _coerce_localized_numbers(df)
    return _coerce_datetimes(df)


def _text_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if pd.api.types.is_string_dtype(df[column])]


def _sample(series: pd.Series) -> pd.Series:
    return series.dropna().astype(str).str.strip().head(_SAMPLE_SIZE)


def _coerce_localized_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas textuais em formato numérico brasileiro (ex.: 1.234,56)."""
    for column in _text_columns(df):
        sample = _sample(df[column])
        if sample.empty or not sample.str.contains(",", regex=False).any():
            continue
        if sample.str.match(_LOCALIZED_NUMBER).mean() < _PARSE_THRESHOLD:
            continue
        cleaned = (
            df[column]
            .str.strip()
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df[column] = pd.to_numeric(cleaned, errors="coerce")
        logger.debug("Coluna '%s' convertida para numérica (formato pt-BR).", column)
    return df


def _to_datetime(series: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=True)


def _coerce_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas textuais que contêm datas em datetime."""
    for column in _text_columns(df):
        sample = _sample(df[column])
        if sample.empty:
            continue
        if pd.to_numeric(sample, errors="coerce").notna().mean() > 0.5:
            continue
        if _to_datetime(sample).notna().mean() >= _PARSE_THRESHOLD:
            df[column] = _to_datetime(df[column])
            logger.debug("Coluna '%s' convertida para datetime.", column)
    return df