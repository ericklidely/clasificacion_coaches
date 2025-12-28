from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd


def add_pca_and_plot_clusters_2d(
    df_features: pd.DataFrame,
    cluster_col: str = "cluster",
    drop_cols: list = None,
    cluster_labels: dict = None,
    title: str = "Segmentación de coaches por nivel de experiencia (PCA)",
    output_path: str = "clusters_pca_2d.png",
    pca_cols: tuple = ("PC1", "PC2")
) -> pd.DataFrame:
    """
    Calcula PCA (2D), agrega PC1 y PC2 al dataframe,
    grafica los clusters y guarda la imagen.

    Returns
    -------
    pd.DataFrame
        DataFrame original + columnas PC1 y PC2
    """

    df = df_features.copy()

    # Columnas para PCA (excluyendo cluster e identificadores)
    cols_to_drop = [cluster_col]
    if drop_cols:
        cols_to_drop += drop_cols

    X = df.drop(columns=cols_to_drop, errors="ignore")

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # Agregar PCA al dataframe
    df[pca_cols[0]] = X_pca[:, 0]
    df[pca_cols[1]] = X_pca[:, 1]

    # ===== Plot =====
    plt.figure(figsize=(10, 7))

    for cluster in sorted(df[cluster_col].unique()):
        subset = df[df[cluster_col] == cluster]

        label = (
            cluster_labels.get(cluster, f"Cluster {cluster}")
            if cluster_labels
            else f"Cluster {cluster}"
        )

        plt.scatter(
            subset[pca_cols[0]],
            subset[pca_cols[1]],
            label=label,
            alpha=0.7
        )

    plt.title(title)
    plt.xlabel("Componente principal 1 (PC1)")
    plt.ylabel("Componente principal 2 (PC2)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.show()

    explained_var = pca.explained_variance_ratio_.sum()
    print(f"Varianza explicada por PCA (2D): {explained_var:.2%}")
    print(f"Gráfico guardado en: {output_path}")

    return df
