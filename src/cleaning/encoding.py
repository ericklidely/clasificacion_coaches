import pandas as pd
from typing import List

def one_hot_encode_columns(df: pd.DataFrame, 
                           cols: List[str], 
                           drop_first: bool = False) -> pd.DataFrame:
    """
    Aplica One-Hot Encoding a columnas categóricas específicas y elimina las originales.

    Parámetros:
        df (pd.DataFrame): DataFrame de entrada.
        cols (List[str]): Lista de columnas a codificar.
        drop_first (bool): Si True, elimina la primera categoría por columna (evita multicolinealidad).

    Retorna:
        pd.DataFrame: DataFrame con columnas codificadas y originales eliminadas.
    """
    df = df.copy()

    # Validar que las columnas existan
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas no encontradas en el DataFrame: {missing}")

    # One-Hot Encoding
    encoded = pd.get_dummies(df[cols], prefix=cols, drop_first=drop_first)

    # Eliminar columnas originales y concatenar las nuevas
    df = df.drop(columns=cols)
    df = pd.concat([df, encoded], axis=1)

    return df
