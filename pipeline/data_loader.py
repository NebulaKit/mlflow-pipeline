import polars as pl
import pandas as pd
from typing import Tuple


def load_data(path: str, label_col: str) -> Tuple[pl.DataFrame, pl.Series]:
    """
    Loads a CSV or TSV file into a pandas DataFrame,
    splits it into features (X) and target (y).

    Parameters:
        path (str): Path to a CSV or TSV file.
        label_col (str): Name of the label/target column.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y)
    """
    if path.lower().endswith(".tsv"):
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y


def load_data_pl(path: str, label_col: str) -> Tuple[pl.DataFrame, pl.Series]:
    """
    Loads a CSV or TSV file into a Polars DataFrame,
    splits it into features (X) and target (y).

    Parameters:
        path (str): Path to a CSV or TSV file.
        label_col (str): Name of the label/target column.

    Returns:
        Tuple[pl.DataFrame, pl.Series]: Features (X) and target (y)
    """
    if path.lower().endswith(".tsv"):
        df = pl.read_csv(path, separator="\t")
    else:
        df = pl.read_csv(path)

    X = df.drop(label_col)
    y = df[label_col]
    return X, y
