import pandas as pd

from src.cleaning.cleaning import *
from src.cleaning.encoding import one_hot_encode_columns
from src.cleaning.embeddings import embed_texts
from src.features.pca import pca_dataframe
#from src.features.silhoutte import find_optimal_k_silhouette
from src.features.cluster import fit_kmeans
from sklearn.metrics import silhouette_score
from src.features.plot_clusters import add_pca_and_plot_clusters_2d

file_path = './data/normalized_data_coaches_tec_v2.xlsx'

df_raw = pd.read_excel(file_path, sheet_name='normalized_data_v4')

df_raw = df_raw[~df_raw['Email'].isin(['nttrevino.entity@gmail.com','octavio.garcia@tec.mx'])]


numeric_cols = ['horas_de_practica','anos_de_experiencia','horas_de_formacion','edad']

drop_cols = ['nombre', 'apellido_paterno', 'apellido_materno', 'pais_de_nacimiento', 'estado_de_residencia', 
             'pais_de_residencia', 'celular', 'fuente', 'tipo_de_contrato', 'curp_/_dni', 
             'foto', 'esta_en_documento_de_semblanzas', 'tipo_de_coaching', 'icf', 'emcc', 'icc', 'wabc', 'assessments', 
             'otros', 'concatenado', 'validacion', 'tipo_de_clientes', 'perfiles_clientes', 'tipo_industria',
             '?esta_en_documento_de_semblanzas?', 'fecha_de_nacimiento', 'genero']

binary_cols = ['ha_sido_coachee', 'cuenta_con_certificacion', 'puedes_brindar_coaching_en_ingles']

df_raw = df_raw.fillna({
    'ha_sido_coachee': 'no',
    'cuenta_con_certificacion': 'no',
    'puedes_brindar_coaching_en_ingles': 'no'
})

encode_col = ['respecto_al_coaching', 'ultima_vez_que_atendio_cliente', 'recibe_supervision_en_practica_como_coach', 'ultima_capacitacion_en_coaching']

df_profile = build_coach_profile_text(df_raw)
df_profile = normalize_column_names(df_profile)
df_profile = clean_text_columns(df_profile)

df_unique = clean_dataset(
    df=df_raw,
    numeric_cols=numeric_cols,
    excel_birthdate_col='fecha_de_nacimiento',
    drop_cols=drop_cols,
    scale_numeric=True,
    id_col='email'
)

df = df_unique.merge(df_profile, on='email', how='left')

df = encode_binary_columns(df, binary_cols=binary_cols)

df = one_hot_encode_columns(df, cols=encode_col)


embeddings = embed_texts(
    texts=df['texto_perfil'],
    backend="sentence_transformers"
)

emb_df = pd.DataFrame(
    embeddings,
    index=df.index,
    columns=[f"emb_{i}" for i in range(embeddings.shape[1])]
)



df_emb_pca, pca_emb = pca_dataframe(emb_df, n_components=0.95, prefix="emb")

X_final = pd.concat([df, df_emb_pca], axis=1)


X_final_fit = X_final.drop(columns=['email','texto_perfil'], errors='ignore')


#best_k, scores = find_optimal_k_silhouette(X_final, k_min=2, k_max=12)

kmeans, labels = fit_kmeans(X_final_fit, n=3)
X_final["cluster"] = labels

X_final = add_pca_and_plot_clusters_2d(
    df_features=X_final,
    cluster_col="cluster",
    drop_cols=["email",'texto_perfil'],
    cluster_labels={
        0: "Senior",
        1: "Junior",
        2: "Mid-level"
    },
    output_path="segmentacion_coaches_pca.png"
)

X_final.to_csv('data/coaches_cluster_pbi.csv', index=False)

X_final.to_csv('/Users/ericklopez/Library/CloudStorage/OneDrive-InstitutoTecnologicoydeEstudiosSuperioresdeMonterrey/Sharepoint VPAF/coaches_cluster/coaches_cluster_pbi.csv', index=False)