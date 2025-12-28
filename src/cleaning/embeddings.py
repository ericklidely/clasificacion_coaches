from typing import List, Literal, Optional, Union
import numpy as np
import pandas as pd

BackendType = Literal["sentence_transformers", "openai"]


def _to_list_of_str(texts: Union[List[str], pd.Series, pd.Index]) -> List[str]:
    """
    Normaliza la entrada a una lista de strings.
    Reemplaza NaN por cadena vacía.
    """
    if isinstance(texts, (pd.Series, pd.Index)):
        return texts.fillna("").astype(str).tolist()
    elif isinstance(texts, list):
        return ["" if x is None else str(x) for x in texts]
    else:
        raise TypeError("texts debe ser list[str], pd.Series o pd.Index")


def embed_texts(
    texts: Union[List[str], pd.Series, pd.Index],
    backend: BackendType = "sentence_transformers",
    st_model_name: str = "all-MiniLM-L6-v2",
    openai_model: str = "text-embedding-3-small",
    openai_client: Optional[object] = None,
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Genera embeddings a partir de una lista/Serie de textos usando el backend elegido.

    Parámetros
    ----------
    texts : list[str] | pd.Series | pd.Index
        Textos a convertir en embeddings.
    backend : {"sentence_transformers", "openai"}
        Modelo a utilizar.
    st_model_name : str
        Nombre del modelo de SentenceTransformers (si backend="sentence_transformers").
    openai_model : str
        Nombre del modelo de OpenAI Embeddings (si backend="openai").
    openai_client : OpenAI | None
        Instancia del cliente OpenAI (si no se pasa, se crea una con la API key de entorno).
    batch_size : int
        Tamaño de batch para sentence_transformers.
    show_progress : bool
        Mostrar barra de progreso en sentence_transformers.

    Returns
    -------
    np.ndarray
        Matriz (n_samples, embedding_dim) con los embeddings.
    """
    texts_list = _to_list_of_str(texts)

    if backend == "sentence_transformers":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "No se encontró el paquete 'sentence_transformers'. "
                "Instálalo con: pip install sentence-transformers"
            ) from e

        model = SentenceTransformer(st_model_name)
        embeddings = model.encode(
            texts_list,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        )
        return np.array(embeddings, dtype="float32")

    elif backend == "openai":
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "No se encontró el paquete 'openai'. "
                "Instálalo con: pip install openai"
            ) from e

        client = openai_client or OpenAI()  # usa OPENAI_API_KEY del entorno
        # Llamada batcheada por si la lista es larga
        embeddings_list: List[List[float]] = []

        # OpenAI permite enviar una lista completa en 'input', pero si quieres
        # ser más defensivo, puedes batch-ear aquí:
        response = client.embeddings.create(
            model=openai_model,
            input=texts_list,
        )
        for item in response.data:
            embeddings_list.append(item.embedding)

        return np.array(embeddings_list, dtype="float32")

    else:
        raise ValueError(f"backend no soportado: {backend}")
