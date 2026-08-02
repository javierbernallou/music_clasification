import io

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import  NearestNeighbors



st.set_page_config(page_title="Music Similarity Analysis", layout="wide")


st.title("Music EDA: cómo describir y comparar canciones con pocas dimensiones")

st.markdown(
    """
Spotify (y cualquier plataforma de streaming musical) describe cada canción con un conjunto de
**características de audio**: energía, bailabilidad, tonalidad, tempo, positividad emocional (valence),
grado acústico, etc. Son 15 variables numéricas por canción, cada una capturando un matiz distinto del sonido.

El problema es que **15 variables son demasiadas para razonar visualmente o comparar canciones de forma
directa**, y muchas de ellas están correlacionadas entre sí (una canción con mucha *energy* casi siempre
tiene mucho *loudness*, por ejemplo). Este proyecto resuelve ese problema en cuatro pasos:

1. **Exploración (EDA):** entender la calidad y distribución de los datos.
2. **Reducción de dimensionalidad (PCA):** comprimir las 15 variables originales en un puñado de
   "super-variables" (componentes principales) que conservan la mayor parte de la información.
3. **Motor de similitud:** una vez las canciones viven en ese espacio reducido, usar **similitud de
   coseno** para responder la pregunta de negocio real: *dada una canción, ¿cuáles se le parecen más?*
   Esta es, en esencia, la lógica detrás de un sistema de recomendación tipo "Radio" o "Fans también
   escuchan".
"""
)

st.sidebar.header("Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu propio CSV ", type=["csv"])
st.sidebar.caption("Si no subes nada, se usará `dataset.csv` en el directorio de la app.")


@st.cache_data(show_spinner="Cargando dataset...")
def load_data(file_bytes: bytes = None, local_path: str = "dataset.csv") -> pd.DataFrame:
    if file_bytes is not None:
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_csv(local_path)


try:
    df_raw = load_data(uploaded_file.getvalue() if uploaded_file is not None else None)
except Exception as e:
    st.error(
        f"No se pudo cargar el dataset ({e}). Sube un CSV desde la barra lateral o coloca "
        f"`dataset.csv` junto a la app."
    )
    st.stop()

df = df_raw.copy()
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

n_before = len(df)
df = df.dropna().copy()


n_after = len(df)

st.header("1) Panorama general del dataset")
c1, c2, c3 = st.columns(3)
c1.metric("Canciones cargadas", f"{n_before:,}")
c2.metric("Canciones tras limpiar nulos", f"{n_after:,}", delta=f"-{n_before - n_after:,}")
c3.metric("Columnas", f"{df.shape[1]}")

st.markdown(
    """
Cada fila representa una canción etiquetada con un género (una misma canción puede aparecer varias
veces si Spotify la asocia a más de un género). Antes de tocar los datos, revisamos su forma y
descartamos filas incompletas para que ni el PCA ni el resto de cálculos se vean distorsionados por
valores faltantes.
"""
)
st.dataframe(df.head(10), width="stretch")

feature_cols = [
    "popularity", "duration_ms", "explicit", "danceability", "energy", "key",
    "loudness", "mode", "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "time_signature"
]
numeric_cols = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
numeric_cols = [c for c in numeric_cols if df[c].dtype != bool]


st.header("2) Filtro de géneros")
all_genres = sorted(df["track_genre"].dropna().astype(str).unique()) if "track_genre" in df.columns else []
default_genres = [
    "pop", "k-pop", "reggae", "reggaeton", "rock", "hard-rock", "punk-rock", "brazil",
    "ambient", "alternative", "blues", "classical", "club", "dance", "disco",
    "drum-and-bass", "edm", "folk", "gospel", "hardcore", "hardstyle",
    "heavy-metal", "hip-hop", "indie", "house", "indie-pop", "jazz", "latino",
    "metal", "opera", "romance", "sad", "salsa", "spanish", "study", "techno"
]

selected_genres = all_genres
if all_genres:
    selected_genres = st.multiselect(
        "Géneros a conservar (afecta a toda la app: EDA, PCA y motor de similitud)",
        options=all_genres,
        default=[g for g in default_genres if g in all_genres] or all_genres[: min(20, len(all_genres))]
    )
    if selected_genres:
        df = df[df["track_genre"].astype(str).isin(selected_genres)].copy()
    st.caption(f"Géneros disponibles: {len(all_genres)} | Géneros seleccionados: {len(selected_genres)}")

if df.empty:
    st.warning("No quedan canciones tras aplicar el filtro de géneros. Selecciona al menos un género.")
    st.stop()

df = df.drop_duplicates(subset=["track_name", "artists"], keep="first").reset_index(drop=True)#ELIMINAMOS CANCIONES DUPLICADAS CON DIFERENTES GÉNEROS


st.header("3) Distribución de las variables acústicas")
st.markdown(
    "Antes de reducir dimensiones, conviene ver qué géneros destacan en las variables más intuitivas: "
    "**popularidad** y **bailabilidad**."
)

col_pop, col_dance = st.columns(2)

with col_pop:
    top_pop = df.groupby("track_genre")["popularity"].mean().reset_index().nlargest(10, "popularity")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x="popularity", y="track_genre", data=top_pop, ax=ax, hue="track_genre",
                palette="flare", legend=False)
    ax.set_title("Top 10 géneros más populares")
    ax.set_xlabel("Popularidad media")
    ax.set_ylabel("")
    st.pyplot(fig, clear_figure=True)

with col_dance:
    top_dance = df.groupby("track_genre")["danceability"].mean().reset_index().nlargest(10, "danceability")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x="danceability", y="track_genre", data=top_dance, ax=ax, hue="track_genre",
                palette="crest", legend=False)
    ax.set_title("Top 10 géneros más bailables")
    ax.set_xlabel("Danceability media")
    ax.set_ylabel("")
    st.pyplot(fig, clear_figure=True)

st.caption(
    f"'{top_pop.iloc[0]['track_genre']}' lidera en popularidad media y "
    f"'{top_dance.iloc[0]['track_genre']}' en bailabilidad media dentro de la selección actual de géneros."
)

st.header("4) Relación entre variables")
corr_cols = [c for c in numeric_cols if c in df.columns]
if corr_cols:
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Matriz de correlación de variables acústicas")
    st.pyplot(fig, clear_figure=True)

    # Par más correlacionado (excluyendo la diagonal) para dar contexto dinámico
    corr_abs = corr.abs().where(~np.eye(len(corr), dtype=bool))
    max_pair = corr_abs.stack().idxmax()
    max_val = corr.loc[max_pair]

    st.markdown(
        f"""
Esta matriz muestra qué variables se mueven juntas. El par más correlacionado dentro de la selección
actual es **{max_pair[0]}** y **{max_pair[1]}** (r = {max_val:.2f}). Cuando dos variables cuentan casi
la misma historia, son buenas candidatas a fundirse en una sola componente durante el PCA: esa
redundancia es precisamente lo que la reducción de dimensionalidad va a aprovechar.
"""
    )


st.header("5) Escalado de variables")
pca_cols = [c for c in numeric_cols if c in df.columns]

if len(pca_cols) < 2:
    st.error("No hay suficientes variables numéricas para continuar con el análisis.")
    st.stop()

st.markdown(
    """
Las variables de audio viven en escalas muy distintas (`tempo` puede rondar 50-220, mientras que
`valence` o `danceability` van de 0 a 1). Si no se estandarizan, el PCA quedaría dominado por las
variables con mayor rango numérico simplemente por su magnitud, no porque aporten más información.
Por eso primero se aplica `StandardScaler`, que transforma cada variable a media 0 y desviación
estándar 1.
"""
)

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[pca_cols]), columns=pca_cols, index=df.index)
scaled_values = df_scaled.values

sample_for_cluster = df_scaled
if len(sample_for_cluster) > 5000:
    sample_for_cluster = sample_for_cluster.sample(n=5000, random_state=42)


@st.cache_data(show_spinner="Buscando el número óptimo de clusters...")
def evaluate_k(sample_values: np.ndarray, k_min: int = 2, k_max: int = 12):
    k_max = min(k_max, max(k_min + 1, len(sample_values) - 1))
    k_range = list(range(k_min, k_max + 1))
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(sample_values)
        inertias.append(km.inertia_)
        try:
            silhouettes.append(silhouette_score(sample_values, labels))
        except Exception:
            silhouettes.append(np.nan)
    return k_range, inertias, silhouettes


st.header("6) Número de clusters sugerido")
st.markdown(
    "Antes de comprimir con PCA, comprobamos si las canciones forman agrupaciones naturales (clusters) "
    "usando el método del codo y el coeficiente de silueta sobre las variables ya escaladas."
)

k_range, inertias, silhouettes = evaluate_k(sample_for_cluster.values)
best_k_silhouette = k_range[int(np.nanargmax(silhouettes))]

col_elbow, col_sil = st.columns(2)
with col_elbow:
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(k_range, inertias, "r-*")
    ax.set_xlabel("K")
    ax.set_ylabel("Inercia")
    ax.set_title("Método del codo")
    ax.grid(color="gray", linewidth=1, axis="y", alpha=0.4)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    st.pyplot(fig, clear_figure=True)

with col_sil:
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(k_range, silhouettes, "b-*")
    ax.axvline(x=best_k_silhouette, color="red", linestyle="--", label=f"Mejor K = {best_k_silhouette}")
    ax.set_xlabel("K")
    ax.set_ylabel("Coeficiente de silueta")
    ax.set_title("Coeficiente de silueta")
    ax.legend()
    ax.grid(color="gray", linewidth=1, axis="y", alpha=0.4)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    st.pyplot(fig, clear_figure=True)

st.markdown(
    f"""
El coeficiente de silueta más alto se da en **K = {best_k_silhouette}**. Un valor de K bajo es
razonable: muchos géneros comparten rasgos acústicos (el pop y el reggaetón, por ejemplo, se solapan
en bailabilidad y energía), así que no existe una segmentación perfectamente limpia por clusters. Esto
también anticipa que un espacio de 2-3 dimensiones probablemente se quede corto para separar bien
todos los géneros, algo que confirmaremos más abajo con el mapa 3D.
"""
)


st.header("7) PCA: reducción de dimensionalidad")
st.markdown(
    """
Aplicamos ahora el **Análisis de Componentes Principales (PCA)**. La idea es crear nuevas variables
(componentes) que sean combinaciones lineales de las originales, ordenadas de forma que la primera
capture la mayor cantidad posible de varianza (información), la segunda la mayor cantidad de varianza
restante, y así sucesivamente. Si con pocas componentes conservamos casi toda la varianza, significa
que las 15 variables originales eran en gran parte redundantes entre sí.
"""
)


@st.cache_resource(show_spinner="Ajustando PCA...")
def fit_pca(values: np.ndarray, variance_target: float = 0.95, seed: int = 42):
    model = PCA(n_components=variance_target, random_state=seed)
    transformed = model.fit_transform(values)
    return model, transformed


@st.cache_resource(show_spinner="Ajustando PCA completo (para la curva de varianza)...")
def fit_pca_full(values: np.ndarray, seed: int = 42):
    model = PCA(n_components=None, random_state=seed)
    model.fit(values)
    return model


pca_model, X_pca = fit_pca(scaled_values, variance_target=0.95)
n_components = X_pca.shape[1]
pc_cols = [f"PC{i}" for i in range(1, n_components + 1)]

weights_df = pd.DataFrame(pca_model.components_, columns=df_scaled.columns, index=pc_cols)
explained = pd.Series(pca_model.explained_variance_ratio_, index=pc_cols, name="Varianza de PC")
weights_and_variance = pd.concat([weights_df, explained], axis=1)

st.success(
    f"PCA conserva el **{pca_model.explained_variance_ratio_.sum():.1%}** de la varianza total usando "
    f"**{n_components} componentes** en lugar de las {len(pca_cols)} variables originales "
    f"({n_components / len(pca_cols):.0%} de las columnas originales)."
)


st.subheader("8) Qué explica cada componente")
fig, ax = plt.subplots(figsize=(14, max(4, 0.4 * n_components)))
sns.heatmap(
    weights_and_variance, annot=True, cmap="coolwarm", fmt=".2f",
    linewidths=0.5, cbar=True, ax=ax
)
ax.set_title("Mapa de calor de los pesos del PCA")
ax.set_xlabel("Variables")
ax.set_ylabel("Componentes")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig, clear_figure=True)


def top_features_for_pc(pc_name: str, k: int = 3) -> str:
    row = weights_df.loc[pc_name]
    top = row.reindex(row.abs().sort_values(ascending=False).index).head(k)
    parts = [f"{'+' if v > 0 else '−'}{name}" for name, v in top.items()]
    return ", ".join(parts)


st.markdown("**Lectura automática de cada componente** (variables con más peso, en valor absoluto):")
interpretation_lines = "\n".join(
    f"- **{pc}** ({explained[pc]:.1%} de la varianza): {top_features_for_pc(pc)}"
    for pc in pc_cols
)
st.markdown(interpretation_lines)
st.caption(
    "El signo indica el sentido de la relación: un signo '+' significa que la componente crece cuando "
    "esa variable crece; un '−', lo contrario. Esta lista se genera automáticamente a partir de los "
    "pesos reales, así que se adapta si cambias el filtro de géneros."
)



st.subheader("9) ¿Compensa quedarnos con tantos componentes?")
st.markdown(
    """
El 95% de varianza es un umbral habitual, pero no es el único válido. A continuación se muestra la
**varianza explicada acumulada** en función del número de componentes, para decidir con datos si
compensa relajar ese umbral a cambio de un espacio más compacto (más rápido de indexar, más fácil de
visualizar, menos ruido).
"""
)

pca_full_model = fit_pca_full(scaled_values)
cum_var = np.cumsum(pca_full_model.explained_variance_ratio_)

thresholds = [0.80, 0.85, 0.90, 0.95]
rows = []
for t in thresholds:
    k_needed = int(np.searchsorted(cum_var, t) + 1)
    rows.append({"Varianza objetivo": f"{t:.0%}", "Componentes necesarios": k_needed,
                 "Columnas ahorradas": len(pca_cols) - k_needed})
tradeoff_df = pd.DataFrame(rows)

col_scree, col_table = st.columns([2, 1])
with col_scree:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x_axis = np.arange(1, len(cum_var) + 1)
    ax.bar(x_axis, pca_full_model.explained_variance_ratio_, alpha=0.4, color="steelblue",
           label="Varianza individual")
    ax.plot(x_axis, cum_var, "o-", color="darkred", label="Varianza acumulada")
    ax.axhline(0.95, color="gray", linestyle="--", linewidth=1)
    ax.axhline(0.90, color="gray", linestyle=":", linewidth=1)
    ax.axvline(n_components, color="green", linestyle="--", linewidth=1,
               label=f"Elegido ({n_components} PCs)")
    ax.set_xlabel("Número de componentes")
    ax.set_ylabel("Proporción de varianza")
    ax.set_title("Scree plot: varianza individual vs. acumulada")
    ax.legend(fontsize=8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    st.pyplot(fig, clear_figure=True)

with col_table:
    st.dataframe(tradeoff_df, width="stretch", hide_index=True)

k90 = int(tradeoff_df.loc[tradeoff_df["Varianza objetivo"] == "90%", "Componentes necesarios"].iloc[0])
k95 = int(tradeoff_df.loc[tradeoff_df["Varianza objetivo"] == "95%", "Componentes necesarios"].iloc[0])
ahorro = k95 - k90

st.markdown(
    f"""
**Lectura de la curva:** las primeras componentes concentran mucha información y luego la curva se
aplana (rendimientos decrecientes), lo cual es la señal clásica de que sí existe redundancia entre las
variables originales. En este dataset, pasar del **90% ({k90} PCs)** al **95% ({k95} PCs)** de varianza
cuesta **{ahorro} componente(s) adicionales**.

- Si el objetivo es **visualizar o explorar** el espacio (radar, mapas 2D/3D), **{k90} componentes al
  90%** suelen ser suficientes y más manejables.
- Si el objetivo es un **motor de similitud/recomendación** donde la precisión importa más que la
  interpretabilidad visual, mantener el **95% ({k95} PCs)** es preferible: se pierde menos matiz al
  comparar canciones por coseno, y {k95} columnas siguen siendo una fracción pequeña de las
  {len(pca_cols)} originales.

Por eso el resto de la app usa el espacio al 95% para el motor de recomendación, pero las gráficas de
validación de abajo comprueban si de verdad hace falta tanta precisión para separar géneros.
"""
)



n_unique_genres = df["track_genre"].nunique()



st.subheader("11) Mapa 3D de canciones por género")
st.markdown(
    """
El siguiente mapa proyecta cada canción sobre sus tres primeras componentes principales. Sirve como
"prueba visual" de la reducción de dimensionalidad: si los puntos de un mismo género tienden a
agruparse, significa que esas 3 dimensiones (de las 15 originales) ya capturan parte real de lo que
distingue a un género de otro.
"""
)


@st.cache_data(show_spinner="Proyectando en 3 componentes...")
def compute_pca_3d(values: np.ndarray, seed: int = 42):
    model = PCA(n_components=3, random_state=seed)
    return model.fit_transform(values)


if "track_genre" in df.columns:
    X_pca_3d = compute_pca_3d(scaled_values)
    plot_df_3d = pd.DataFrame(X_pca_3d, columns=["PC1", "PC2", "PC3"], index=df.index)
    plot_df_3d["track_genre"] = df["track_genre"].astype(str).values
    plot_df_3d["track_name"] = df["track_name"].astype(str).values

    fig = px.scatter_3d(
        plot_df_3d, x="PC1", y="PC2", z="PC3", color="track_genre",
        hover_data=["track_name"],
        title="Canciones proyectadas en los tres primeros componentes",
        opacity=0.6
    )
    fig.update_traces(marker=dict(size=3))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": True, "scrollZoom": True})

    st.caption(
        "Con solo 3 de las ~12 dimensiones necesarias para el 95% de varianza ya se aprecian nubes de "
        "puntos por género, aunque con solapamientos — coherente con el K óptimo bajo encontrado antes."
    )


df_pca_full = pd.DataFrame(X_pca, columns=pc_cols, index=df.index)
df_pca_full["artists"] = df["artists"].astype(str).values
df_pca_full["track_name"] = df["track_name"].astype(str).values
df_pca_full["track_genre"] = df["track_genre"].astype(str).values

st.subheader("12) Perfil radial por género")
st.markdown(
    """
El siguiente gráfico muestra el **perfil medio de cada género** en las componentes principales. Cada eje del gráfico radial representa una componente, 
y la distancia desde el centro indica el valor medio de esa componente.
"""
)
chosen_genre = st.selectbox("Elige un género para el gráfico radial", sorted(df["track_genre"].dropna().astype(str).unique()))
genre_subset = df_pca_full[df_pca_full["track_genre"] == chosen_genre]

if not genre_subset.empty:
    valores = genre_subset[pc_cols].mean().values
    fig = px.line_polar(r=valores, theta=pc_cols, markers=True, title=f"Gráfico radial — Género: {chosen_genre}")
    fig.update_traces(fill="toself")
    st.plotly_chart(fig, width="stretch")
else:
    st.info("No hay filas para ese género en la selección actual.")

st.subheader("13) Un artista frente a su género promedio")
st.markdown(
    """
Esta comparación responde a una pregunta distinta: **¿un artista suena "típico" de su género o se
sale de la norma?** Se calcula el perfil medio de todas las canciones del artista y se superpone al
perfil medio del género que más veces se le asocia.
"""
)


@st.cache_data(show_spinner=False)
def build_artist_index(artists_series: pd.Series) -> pd.DataFrame:
    """Devuelve una fila por cada pareja (índice original, artista individual)."""
    exploded = artists_series.astype(str).str.split(";").explode().str.strip()
    exploded = exploded[exploded != ""]
    return pd.DataFrame({"row_index": exploded.index, "artist": exploded.values})


artist_index = build_artist_index(df["artists"])
artists_available = sorted(artist_index["artist"].unique())

if artists_available:
    artist_for_radar = st.selectbox("Elige un artista", artists_available, key="radar_artist")
    rows_artist = artist_index.loc[artist_index["artist"] == artist_for_radar, "row_index"].unique()
    artist_pcs = df_pca_full.loc[df_pca_full.index.intersection(rows_artist)]

    if not artist_pcs.empty:
        main_genre = artist_pcs["track_genre"].mode().iat[0]
        artist_profile = artist_pcs[pc_cols].mean()
        genre_profile = df_pca_full.loc[df_pca_full["track_genre"] == main_genre, pc_cols].mean()

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=artist_profile.values, theta=pc_cols, fill="toself", name=artist_for_radar
        ))
        fig.add_trace(go.Scatterpolar(
            r=genre_profile.values, theta=pc_cols, fill="toself", name=f"Promedio '{main_genre}'", opacity=0.6
        ))
        fig.update_layout(
            title=f"{artist_for_radar} vs. promedio de '{main_genre}'",
            showlegend=True
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"{artist_for_radar} aparece con {len(artist_pcs)} canción(es) en la selección actual, "
            f"etiquetadas mayoritariamente como '{main_genre}'."
        )
    else:
        st.info("No hay canciones de ese artista en la selección actual de géneros.")
else:
    st.info("No se encontraron artistas individuales en la columna `artists`.")


st.subheader("14) Comparación 3D entre artistas")
st.markdown(
    """
A continuación, se pueden comparar 2 artistas en un gráfico en 3 dimensiones, eligiendo los 3 componentes en los cuales quieres compararlos.
"""
)

MAX_POINTS_PER_ARTIST = 300

if len(artists_available) >= 2:
    default_a2 = artists_available[1] if len(artists_available) > 1 else artists_available[0]
    chosen_artists = st.multiselect(
        "Elige hasta 2 artistas para comparar",
        options=artists_available,
        default=[artists_available[0], default_a2],
        max_selections=2,
    )
    chosen_pcs = st.multiselect(
        "Elige 3 componentes para los ejes X, Y, Z",
        options=pc_cols,
        default=pc_cols[:3],
        max_selections=3,
    )

    if len(chosen_artists) >= 1 and len(chosen_pcs) == 3:
        matches = artist_index[artist_index["artist"].isin(chosen_artists)]
        match_labels = matches.groupby("row_index")["artist"].apply(lambda s: " & ".join(sorted(set(s))))

        valid_idx = df_pca_full.index.intersection(match_labels.index)
        subset = df_pca_full.loc[valid_idx].copy()
        subset["artista_seleccionado"] = match_labels.loc[valid_idx]


        if len(subset) > 0:
            sampled_chunks = [
                grupo.sample(min(len(grupo), MAX_POINTS_PER_ARTIST), random_state=42)
                for _, grupo in subset.groupby("artista_seleccionado")
            ]
            subset = pd.concat(sampled_chunks, axis=0)

        if not subset.empty:
            fig = px.scatter_3d(
                subset, x=chosen_pcs[0], y=chosen_pcs[1], z=chosen_pcs[2],
                color="artista_seleccionado",
                hover_data=["track_name", "track_genre"],
                title=f"Comparación en PCA 3D: {' vs '.join(chosen_artists)}",
                opacity=0.7
            )
            fig.update_traces(marker=dict(size=4))
            st.plotly_chart(fig, width="stretch")
            st.caption(
                f"Mostrando hasta {MAX_POINTS_PER_ARTIST} canciones por artista para mantener la "
                f"gráfica fluida ({len(subset)} puntos dibujados en total)."
            )
        else:
            st.info("No hay canciones de esos artistas en la selección actual de géneros.")
    else:
        st.info("Selecciona al menos un artista y exactamente 3 componentes.")
else:
    st.info("No hay suficientes artistas individuales para comparar en la selección actual.")

st.header("15) Motor de recomendación: canciones similares por similitud de coseno")
st.markdown(
    """
A continuación, a partir de una cancion dada por el usuario, se mostrarán canciones que son similares en el espacio que hemos creado. Esto se hará mediante la **similitud del coseno**.
Esto, lo que hace es definir que tan parecidos son dos vectores (en este caso canciones), al observar el ángulo entre estos.
"""
)

song_labels = df_pca_full["track_name"] + " — " + df_pca_full["artists"]
label_to_index = pd.Series(df_pca_full.index.values, index=song_labels.values)

chosen_label = st.selectbox("Elige una canción de referencia", options=sorted(song_labels.unique()))
top_n = st.slider("Número de canciones similares a mostrar", min_value=3, max_value=25, value=10)

if chosen_label:
    chosen_idx = label_to_index[chosen_label]
    if isinstance(chosen_idx, pd.Series):
        chosen_idx = chosen_idx.iloc[0]

    position = df_pca_full.index.get_loc(chosen_idx)
    query_vector = X_pca[position].reshape(1, -1)
    similarities = cosine_similarity(query_vector, X_pca).flatten()

    result_df = df_pca_full[["track_name", "artists", "track_genre"]].copy()
    result_df["similitud"] = similarities
    result_df["song_label"] = result_df["track_name"] + " — " + result_df["artists"]
    result_df = result_df[result_df["song_label"] != chosen_label].copy()
    result_df = result_df.sort_values("similitud", ascending=False).head(top_n)
    final_df = result_df[["track_name", "artists", "track_genre", "similitud"]].copy()

    st.dataframe(
        final_df.style.format({"similitud": "{:.3f}"}).background_gradient(subset=["similitud"], cmap="Greens"),
        width="stretch"
    )

    top1_idx = result_df.index[0]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=df_pca_full.loc[chosen_idx, pc_cols].values, theta=pc_cols, fill="toself",
        name=chosen_label.split(" — ")[0]
    ))
    fig.add_trace(go.Scatterpolar(
        r=df_pca_full.loc[top1_idx, pc_cols].values, theta=pc_cols, fill="toself",
        name=df_pca_full.loc[top1_idx, "track_name"], opacity=0.6
    ))
    fig.update_layout(title="Canción elegida vs. su recomendación más parecida", showlegend=True)
    st.plotly_chart(fig, width="stretch")


st.header("16) Motor de recomendación: canciones similares por Nearest Neighbors")
st.markdown("""
Tras comprobar cómo la similitud del coseno permite medir el parecido entre canciones en el espacio generado por el PCA, el siguiente paso consiste en utilizar un algoritmo
 diseñado específicamente para realizar este tipo de búsquedas de forma eficiente: Nearest Neighbors. Mientras que en el apartado anterior se calculaba la similitud entre 
 la canción seleccionada y todas las demás para ordenarlas posteriormente, este algoritmo indexa previamente el conjunto de datos y devuelve directamente las canciones más próximas 
 según la métrica elegida. En este caso se mantiene la distancia del coseno como criterio de comparación, por lo que el concepto de similitud no cambia; 
 únicamente cambia la estrategia utilizada para localizar las canciones más parecidas, ofreciendo una solución más escalable cuando el número de canciones aumenta.
""")

X = df_pca_full[pc_cols].values
nn = NearestNeighbors(n_neighbors=(top_n + 1), metric="cosine")
nn.fit(X)

distancias, indices = nn.kneighbors(X[chosen_idx].reshape(1, -1))

recomendaciones_nn = df_pca_full.iloc[indices.flatten()][["track_name", "artists", "track_genre"]].copy()
recomendaciones_nn["similitud"] = 1 -distancias[0]
recomendaciones_nn = recomendaciones_nn[recomendaciones_nn.index != chosen_idx].head(top_n)
recomendaciones_nn = recomendaciones_nn.sort_values("similitud", ascending=False)
st.dataframe(
        recomendaciones_nn.style.format({"similitud": "{:.3f}"}).background_gradient(subset=["similitud"], cmap="Greens"),
        width="stretch"
    )


st.header("17) Conclusiones y cómo continuar")
st.markdown(
    f"""
**Lo que confirma este proyecto:**
- Se ha podido reducir la dimensionalidad de las 15 variables iniciales, quedandonos al final con 12, pero con posibilidad de aumentar su reducción.
- La estructura de géneros es visible incluso en solo 2-3 dimensiones, lo que valida que el espacio reducido no es solo una compresión, sino que conserva
  significado musical real.
- La similitud de coseno sobre ese espacio permite construir recomendaciones razonables sin necesidad
  de las 15 columnas originales.

"""
)