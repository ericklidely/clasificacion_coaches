# src/cleaning/cleaning.py

import pandas as pd
from unidecode import unidecode
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler


# 2) Normalización de nombres de columnas -----------------------------------

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pasa nombres de columnas a minúsculas, sin acentos y con _
    """
    df = df.copy()
    df.columns = [
        unidecode(col).strip().lower().replace(" ", "_").replace('?','').replace('¿','')
        for col in df.columns
    ]
    return df


# 3) Limpieza de texto en columnas categóricas ------------------------------

def clean_text_columns(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convierte a string, minúsculas y elimina acentos en columnas de texto.
    Si cols es None, actúa sobre todas las columnas object/string.
    """
    df = df.copy()
    if cols is None:
        cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    
    for col in cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .apply(lambda x: unidecode(x.lower()))
            .replace({"nan": pd.NA})
        )
    return df


#3) Agrupar las variables categóricas para hacer perfil único
# 3) Agrupar variables categóricas para crear perfil textual semántico

def build_coach_profile_text(
    df,
    industrias_col: str = "Tipo industria",
    id_col: str = "Email"
) -> pd.DataFrame:
    """
    Construye un perfil textual del coach basado únicamente
    en los tipos de industria, ordenados alfabéticamente,
    optimizado para embeddings.
    """
    df_aux = df.copy()

    # Agrupar por coach y obtener industrias únicas ordenadas
    agg = (
        df_aux
        .groupby(id_col)
        .agg({
            industrias_col: lambda x: sorted({v for v in x.dropna()})
        })
        .reset_index()
    )

    # Construir texto semántico consistente
    agg["texto_perfil"] = (
        "Los tipos de industria en los que tengo experiencia son "
        + agg[industrias_col].apply(
            lambda x: ", ".join(x) if x else "no especificados"
        )
        + "."
    )

    return agg[[id_col, "texto_perfil"]]




# 4) Conversión de tipos básicos (numéricos, fechas) ------------------------

def convert_numeric_columns(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """
    Convierte columnas específicas a numéricas (coerce → NaN si no se puede convertir).
    """
    df = df.copy()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def convert_date_from_excel_serial(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Convierte una columna que viene como número Excel (ej. 24771) a datetime.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(
        pd.to_numeric(df[date_col], errors="coerce"),
        origin="1899-12-30",
        unit="D",
        errors="coerce"
    )
    return df


# 5) Features derivadas (ejemplo: edad a partir de fecha_nacimiento) -------

def add_age_from_birthdate(df: pd.DataFrame, birth_col: str, age_col: str = "edad") -> pd.DataFrame:
    """
    Calcula la edad (en años) a partir de fecha de nacimiento.
    """
    df = df.copy()
    hoy = pd.Timestamp.today()
    df[age_col] = ((hoy - df[birth_col]).dt.days / 365.25).astype("float")
    return df


# 6) Manejo de nulos --------------------------------------------------------

def handle_missing_values(
    df: pd.DataFrame,
    numeric_fill: Optional[float] = None,
    categorical_fill: Optional[str] = None
) -> pd.DataFrame:
    """
    Imputa nulos:
    - Numéricas: con numeric_fill (si se especifica) o mediana.
    - Categóricas: con categorical_fill (si se especifica) o 'sin_dato'.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(include=["object", "string"]).columns

    # Numéricas
    for col in num_cols:
        if numeric_fill is not None:
            df[col] = df[col].fillna(numeric_fill)
        else:
            df[col] = df[col].fillna(df[col].median())

    # Categóricas
    for col in cat_cols:
        if categorical_fill is not None:
            df[col] = df[col].fillna(categorical_fill)
        else:
            df[col] = df[col].fillna("no")

    return df


# 7) Selección de variables para modelar ------------------------------------

def select_model_variables(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    drop_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Separa X (features) e y (target) y elimina columnas que no se usarán.
    """
    df = df.copy()

    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    y = None
    if target_col is not None:
        y = df[target_col].copy()
        X = df.drop(columns=[target_col], errors="ignore")
    else:
        X = df

    return X, y


# 9) Normalizar las varibales numéricas

def scale_numeric_columns(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Escala columnas numéricas usando StandardScaler.
    Devuelve el DF escalado y el scaler para aplicarlo después a nuevos datos.
    """
    df = df.copy()
    scaler = StandardScaler()

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, scaler

# 8) Función orquestadora de limpieza completa ------------------------------

def clean_dataset(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    excel_birthdate_col: Optional[str] = None,
    target_col: Optional[str] = None,
    drop_cols: Optional[List[str]] = None,
    scale_numeric: bool = False,
    id_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Pipeline de limpieza completo:
    - Carga datos
    - Normaliza nombres
    - Limpia texto
    - Convierte numéricas
    - Convierte fecha de nacimiento y crea edad (opcional)
    - Imputa nulos
    - Separa X, y
    """
    df = df.copy()
    # 2) Nombres de columnas
    df = normalize_column_names(df)

    # 3) Limpieza de texto en categóricas
    df = clean_text_columns(df)

    # 5) Fecha de nacimiento → edad (opcional)
    if excel_birthdate_col:
        df = convert_date_from_excel_serial(df, excel_birthdate_col)
        df = add_age_from_birthdate(df, birth_col=excel_birthdate_col, age_col="edad")

    # 4) Numéricas
    if numeric_cols:
        df = convert_numeric_columns(df, numeric_cols)

    # 7) Escalado numérico (opcional)
    scaler = None
    if scale_numeric and numeric_cols:
        df, scaler = scale_numeric_columns(df, numeric_cols)


    df.drop_duplicates(id_col ,inplace=True, ignore_index=True)

    # 7) Selección de variables
    X, y = select_model_variables(df, target_col=target_col, drop_cols=drop_cols)

    # 6) Nulos
    X = handle_missing_values(X)

    return X



# Varibales binarias

def encode_binary_columns(df: pd.DataFrame, binary_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    
    replacements = {
        "si": 1, "sí": 1, "yes": 1, "true": 1, "1": 1,
        "no": 0, "false": 0, "0": 0, "": 0
    }

    for col in binary_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.lower()
            .map(replacements)
            .astype("Int64")
        )
    return df