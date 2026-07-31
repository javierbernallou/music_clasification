
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import re

st.set_page_config(page_title="Music Analysis", layout="wide")

st.title("Music EDA: cómo Spotify organiza la similitud entre canciones")
st.write("""
    Este proyecto explora una base de datos de canciones de Spotify como si fuera un mapa musical: 
    primero revisa la calidad de los datos, después estudia la relación entre variables acústicas, 
    y finalmente comprime la información con PCA para ver si las canciones se parecen entre sí por energía, 
    bailabilidad, tonalidad o popularidad."""
)

st.markdown(
    """
La idea principal es responder una pregunta simple: **¿Spotify puede describir canciones parecidas con unas pocas dimensiones?**  
Para eso el flujo va de lo descriptivo a lo interpretativo:

1. inspección y limpieza,
2. distribución y correlación de variables,
3. escalado y reducción de dimensionalidad,
4. lectura de componentes principales,
5. visualización de canciones y géneros en el espacio PCA.
"""
)

# ----------------------------
# Data loading
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None, local_path="dataset.csv"):
    return pd.read_csv(local_path)


try:
    df = load_data()
except Exception as e:
    st.error(
        "No se pudo cargar el dataset. Sube un CSV desde la barra lateral o coloca `dataset.csv` junto a la app."
    )
    st.stop()

# ----------------------------
# Limpieza inicial de datos.
# ----------------------------
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df = df.dropna().copy()

st.header("1) Primer vistazo al dataset")

st.subheader("Muestras iniciales")
st.dataframe(df.head(10), use_container_width=True)




feature_cols = [
    "popularity", "duration_ms", "explicit", "danceability", "energy", "key",
    "loudness", "mode", "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "time_signature"
]
numeric_cols = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
numeric_cols = [c for c in numeric_cols if df[c].dtype != bool]

# ----------------------------
# Genre filter
# ----------------------------
st.header("4) Filtro de géneros")
all_genres = sorted(df["track_genre"].dropna().astype(str).unique()) if "track_genre" in df.columns else []
default_genres = [
    "pop", "k-pop", "reggae", "reggaeton", "rock", "hard-rock", "punk-rock", "brazil",
    "ambient", "alternative", "blues", "classical", "club", "dance", "disco",
    "drum-and-bass", "edm", "folk", "gospel", "hardcore", "hardstyle",
    "heavy-metal", "hip-hop", "indie", "house", "indie-pop", "jazz", "latino",
    "metal", "opera", "romance", "sad", "salsa", "spanish", "study", "techno"
]
if all_genres:
    selected_genres = st.multiselect(
        "Géneros a conservar",
        options=all_genres,
        default=[g for g in default_genres if g in all_genres] or all_genres[: min(20, len(all_genres))]
    )
    if selected_genres:
        df = df[df["track_genre"].astype(str).isin(selected_genres)].copy()
    st.caption(f"Géneros disponibles: {len(all_genres)} | Géneros seleccionados: {len(selected_genres) if selected_genres else 0}")



# ----------------------------
# Numeric exploration
# ----------------------------
df_genero_top_10=df.groupby("track_genre")["popularity"].mean().reset_index()
df_genero_top_10=df_genero_top_10.nlargest(10,"popularity")
st.header("2) Distribución de las variables acústicas")
fig, ax = plt.subplots(figsize=(14,10))
sns.barplot(x="track_genre",y="popularity" , data= df_genero_top_10)
ax.set_title("Generos más populares")
st.pyplot(fig, clear_figure=True)


df_genero_top_10=df.groupby("track_genre")["danceability"].mean().reset_index()
df_genero_top_10=df_genero_top_10.nlargest(10,"danceability")
st.header("2) Distribución de las variables acústicas")
fig, ax = plt.subplots(figsize=(14,10))
sns.barplot(x="track_genre",y="danceability" , data= df_genero_top_10)
ax.set_title("Generos con más características para bailar")
st.pyplot(fig, clear_figure=True)

# ----------------------------
# Correlation
# ----------------------------
st.header("3) Relación entre variables")
corr_cols = [c for c in numeric_cols if c in df.columns]
if corr_cols:
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Variable Correlation Matrix")
    st.pyplot(fig, clear_figure=True)

    st.markdown(
        """
Este mapa ayuda a leer qué variables se mueven juntas. Por ejemplo, `energy` y `loudness` suelen ir de la mano, mientras que `acousticness` tiende a separarse de las canciones más intensas.  
La lógica del EDA aquí es buscar redundancias: si dos variables cuentan casi la misma historia, PCA puede resumirlas.
        """
    )



# ----------------------------
# Scaling and PCA
# ----------------------------
st.header("5) Escalado y reducción de dimensionalidad")
pca_cols = [c for c in numeric_cols if c in df.columns]

if len(pca_cols) >= 2:
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[pca_cols]), columns=pca_cols, index=df.index)


    st.markdown(
        """
Este es el núcleo del análisis: en vez de mirar todas variables por canción, vamos a intentar reducir su dimensionalidad para que sea mas fácil interpretarlas.  
Si la primera componente concentra energía y volumen, y otra recoge bailabilidad o alegría, entonces Spotify está describiendo canciones en un espacio bastante compacto.
"""
    )

    # K evaluation
    sample_for_cluster = df_scaled
    if len(sample_for_cluster) > 5000:
        sample_for_cluster = sample_for_cluster.sample(n=5000, random_state=42)

    st.subheader("6) Número de clusters sugerido")
    st.markdown("""
    Lo primero que haremos será comprobar el número de clusters (agrupaciones) óptima para explicar los datos:     
""")
    max_k = min(12, max(3, len(sample_for_cluster) - 1))
    K = range(2, max_k + 1)
    inertias = []
    silhouettes = []

    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(sample_for_cluster)
        inertias.append(km.inertia_)
        try:
            silhouettes.append(silhouette_score(sample_for_cluster, labels))
        except Exception:
            silhouettes.append(np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(K), inertias, "r-*")
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    ax.axvline(x=4, color="blue")
    ax.grid(color="gray", linewidth=1, axis="y", alpha=0.4)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    st.pyplot(fig, clear_figure=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(K), silhouettes, "b-*")
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette scores")
    ax.set_title("Silhouette Scores")
    ax.axvline(x=3, color="red")
    ax.grid(color="gray", linewidth=1, axis="y", alpha=0.4)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    st.pyplot(fig, clear_figure=True)

    st.markdown(
        """
        El resultado óptimo son 3 agrupaciones, lo que significa que no hay una segmentación clara por grupos, lo cual tiene sentido ya que hay muchos géneros que comparten rasgos, y una
        canción puede ser considerada de varios géneros a la vez.
"""
    )

    st.markdown(
        """
        En el siguiente mapa en 3 dimensiones se puede ver como se agrupan estos datos, se confirma así que con solo 3 dimensiones no somos capaces de captar correctamente las virtudes y
        diferencias de las diferentes canciones.
"""
    )

    if "track_genre" in df.columns:
        st.subheader("8) Mapa 3D de canciones por género")
        pca_3 = PCA(n_components=3, random_state=42)
        X_pca_3d = pca_3.fit_transform(df_scaled)
        pca_3d_df = pd.DataFrame(X_pca_3d, columns=["PC1", "PC2", "PC3"], index=df.index)

        plot_df = pd.concat(
            [pca_3d_df, df[["track_genre"]].astype(str)],
            axis=1
        )

        fig = px.scatter_3d(
            plot_df,
            x="PC1", y="PC2", z="PC3",
            color="track_genre",
            hover_data=[df["track_name"]],
            title="Canciones proyectadas en los tres componentes")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    A continuación, en vez de buscar agrupar los datos en la cantidad óptima de clusters, vamos a aplicar el Análisis de Componentes Principales (PCA), este, se basa en generar nuevas variables
    que sean capaces de explicar la variabilidad de los datos, reduciendo así, la cantidad de columnas con las que trabajar.     
""")

    st.markdown("""
        En este caso, hemos buscado generar el número de componentes principales necesarios para explicar el 95% de la variabilidad. Para lo cual hacen falta 12 PCs. En la siguiente tabla
        se puede observar como representan tanto positiva como negativamente a las variables originales. Pudiendo simplificarlos en las siguientes características:
            
    """)

    
    pca = PCA(n_components=0.95, random_state=42)
    pca_fit = pca.fit(df_scaled)
    X_pca = pca_fit.fit_transform(df_scaled)
    pca_12_df = pesos = pd.DataFrame(pca.components_, columns=df_scaled.columns, index=[f'PC{i}' for i in range(1, pca.n_components_ + 1)])

    st.success(
        f"PCA conserva el 95% de la varianza usando {X_pca.shape[1]} componentes. "
        f"Varianza explicada acumulada: {pca_fit.explained_variance_ratio_.sum():.4f}"
    )

    st.markdown(
        """
- PC1: Energía e intensidad sonora
- PC2: Bailabilidad y positividad
- PC3: Tonalidad y modo
- PC4: Directo/hablado vs. grabado en estudio
- PC5: Ritmo e instrumentalidad
- PC6: Popularidad y presencia vocal
- PC7: Compás y estructura musical
- PC8: Configuración tonal
- PC9: Duración
- PC10: Instrumentalidad
- PC11: Estado de ánimo positivo
- PC12: Bailabilidad vs. valencia emocional  
"""
    )

    explained = pd.DataFrame(
        pca_fit.explained_variance_ratio_,
        index=pca_12_df.index,
        columns=["Varianza de PC"]
    )
    pesos_y_varianza = pd.concat([pca_12_df, explained], axis=1)

    st.subheader("7) Qué explica cada componente")
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        pesos_y_varianza,
        annot=True,
        cmap="coolwarm",
        fmt=".3f",
        linewidths=0.5,
        cbar=True,
        ax=ax
    )
    ax.set_title("Mapa de calor de los pesos del PCA")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Componentes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)





    # Artist comparison
    st.subheader("9) Comparación entre artistas")
    all_artists = sorted(df["artists"].dropna().astype(str).unique())
    individal_artists= set()
    for e in all_artists:
        for a in e.split(";"):
            cleaned = a.strip()
            if cleaned:
                individal_artists.add(cleaned)

    artists_available= sorted(list(individal_artists))

    if artists_available:
        c1, c2 = st.columns(2)
        p1,p2,p3 = st.columns(3)
        with c1:
            a1 = st.selectbox("Artista 1", artists_available, index=0)
        with c2:
            a2 = st.selectbox("Artista 2", artists_available, index=min(1, len(artists_available)-1))
        with p1:
            p1 = st.selectbox("Componente 1", ["PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12" ], index=min(1, 12))
        with p2:
            p2 = st.selectbox("Componente 2", ["PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12" ], index=min(1, 12))
        with p3:
            p3 = st.selectbox("Componente 3", ["PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12" ], index=min(1, 12))


        df_pca_pro = pd.DataFrame(X_pca, columns= [f"PC{i}" for i in range(1,X_pca.shape[1]+1)], index = df_scaled.index)

        df_compare = pd.concat([df_pca_pro, df[["artists", "track_name", "track_genre"]].astype(str)], axis=1)

        pattern = f"(?:^|;)\\s*({re.escape(a1)}|{re.escape(a2)})\\s*(?:;|$)"
        filtro = df_compare["artists"].str.extract(pattern)
        df_filtrada = df_compare[filtro].copy()

        ####REVISAR Y DECIDIR QUE HACER !!!!! PARA GRÁFICO 3D TE QUEDAS CON EL PCA DE 3 O ELIGES 3 VARIABLES PARA SOBRE LAS QUE MOSTRARLO,
        #CONSIDERO QUE LO MEJOR ES QUE LAS VARIABLES(PCAs SOBRE LAS QUE FUERA EL GRÁFICO LAS DECIDIERA EL USUARIO SERIA ÑLO MEJOR
        
        if not df_compare.empty:
            fig = px.scatter_3d(
                df_compare,
                x=p1, y=p2, z=p3,
                color="artists",
                hover_data=[df_compare["track_genre"], df_compare["track_name"]],
                title=f"Comparación en PCA 3D: {a1} vs {a2}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos suficientes para comparar esos artistas en la muestra actual.")

    # Genre averages
    if "track_genre" in df.columns:
        st.subheader("10) Géneros con mayor media en PC1 y PC2")
        pca_with_genre = pd.concat([pd.DataFrame(X_pca, columns=[f"PC{i}" for i in range(1, X_pca.shape[1] + 1)], index=df.index),
                                    df[["track_genre"]].astype(str)], axis=1)

       

        # Final polar plot with dropdown
        st.subheader("11) Perfil radial por género")
        numeric_for_polar = list(pca_with_genre.select_dtypes(include="number").columns)
        chosen_genre = st.selectbox("Elige un género para el gráfico radial", sorted(df["track_genre"].dropna().astype(str).unique()))

        df_filtrado_generos = pca_with_genre[pca_with_genre["track_genre"] == chosen_genre]
        if not df_filtrado_generos.empty:
            valores = df_filtrado_generos[numeric_for_polar].mean().values
            fig = px.line_polar(
                r=valores,
                theta=numeric_for_polar,
                markers=True,
                title=f"Gráfico radial - Género: {chosen_genre}"
            )
            fig.update_traces(fill="toself")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay filas para ese género en la muestra actual.")

        st.markdown(
            """
El gráfico radial es el cierre más intuitivo: muestra cada género como una firma acústica promedio.  
Si dos polígonos se parecen, sus canciones comparten rasgos; si se separan mucho, su sonido medio es distinto.  
Ese es el objetivo final del EDA: pasar de números sueltos a una lectura musical de similitud.
"""
        )
else:
    st.warning("No hay suficientes columnas numéricas para ejecutar PCA.")