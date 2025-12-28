import pandas as pd
from sklearn.decomposition import PCA

def pca_dataframe(
    df: pd.DataFrame,
    n_components=0.95,
    prefix: str = "pca"
) -> pd.DataFrame:
    """
    Aplica PCA a un DataFrame y regresa otro DataFrame con nombres de columnas.
    n_components puede ser int o varianza explicada (ej. 0.95).
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(df.values)

    cols = [f"{prefix}_{i+1}" for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, index=df.index, columns=cols)

    print(f"PCA {prefix}: {len(cols)} componentes | Varianza explicada total: {pca.explained_variance_ratio_.sum():.2%}")
    return df_pca, pca
