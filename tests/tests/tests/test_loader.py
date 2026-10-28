import pandas as pd
import pytest
from analista_dados.loader import DatasetError, load_dataset


def test_semicolon_csv_with_brazilian_numbers_and_dates(tmp_path):
    path = tmp_path / "vendas.csv"
    path.write_text(
        "data;produto;valor\n"
        "31/01/2024;Caneta;1.234,50\n"
        "15/02/2024;Caderno;99,90\n"
        "01/03/2024;Lápis;5,00\n",
        encoding="utf-8",
    )
    df = load_dataset(path)
    assert pd.api.types.is_numeric_dtype(df["valor"])
    assert df["valor"].tolist() == [1234.5, 99.9, 5.0]
    assert pd.api.types.is_datetime64_any_dtype(df["data"])
    assert df["data"].iloc[0] == pd.Timestamp("2024-01-31")


def test_missing_file_raises_dataset_error(tmp_path):
    with pytest.raises(DatasetError, match="não encontrado"):
        load_dataset(tmp_path / "nao_existe.csv")


def test_unsupported_extension_raises_dataset_error(tmp_path):
    path = tmp_path / "dados.xyz"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(DatasetError, match="não suportado"):
        load_dataset(path)


def test_empty_dataset_raises_dataset_error(tmp_path):
    path = tmp_path / "vazio.csv"
    path.write_text("a,b\n")
    with pytest.raises(DatasetError, match="não contém linhas"):
        load_dataset(path)