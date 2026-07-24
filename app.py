from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="EDA de gatos",
    page_icon="🐈",
    layout="wide",
)


BREEDS = [
    "Mestizo",
    "Siamés",
    "Persa",
    "Maine Coon",
    "Bengalí",
    "British Shorthair",
    "Ragdoll",
    "Sphynx",
]

COLORS = [
    "Negro",
    "Blanco",
    "Gris",
    "Naranja",
    "Carey",
    "Atigrado",
    "Bicolor",
]

ACTIVITY_LEVELS = ["Baja", "Media", "Alta"]
SEXES = ["Hembra", "Macho"]

BASE_WEIGHT_BY_BREED = {
    "Mestizo": 4.2,
    "Siamés": 3.8,
    "Persa": 4.5,
    "Maine Coon": 6.8,
    "Bengalí": 5.2,
    "British Shorthair": 5.8,
    "Ragdoll": 6.0,
    "Sphynx": 4.0,
}


@st.cache_data
def generate_cat_data(n_rows: int, seed: int) -> pd.DataFrame:
    """Generate a reproducible synthetic dataset about cats."""
    rng = np.random.default_rng(seed)

    breeds = rng.choice(BREEDS, size=n_rows)
    sexes = rng.choice(SEXES, size=n_rows)
    colors = rng.choice(COLORS, size=n_rows)
    activity = rng.choice(
        ACTIVITY_LEVELS,
        size=n_rows,
        p=[0.25, 0.50, 0.25],
    )

    age_years = np.round(
        np.clip(rng.gamma(shape=2.2, scale=2.6, size=n_rows), 0.2, 18),
        1,
    )

    base_weights = np.array([BASE_WEIGHT_BY_BREED[breed] for breed in breeds])
    sex_adjustment = np.where(sexes == "Macho", 0.45, -0.15)
    age_adjustment = np.minimum(age_years, 5) * 0.08

    weight_kg = np.round(
        np.clip(
            base_weights
            + sex_adjustment
            + age_adjustment
            + rng.normal(0, 0.65, n_rows),
            1.8,
            11.5,
        ),
        2,
    )

    activity_score = (
        pd.Series(activity)
        .map({"Baja": -0.7, "Media": 0.0, "Alta": 0.8})
        .to_numpy()
    )

    sleep_hours = np.round(
        np.clip(
            15.5
            - activity_score
            + age_years * 0.07
            + rng.normal(0, 1.1, n_rows),
            10,
            21,
        ),
        1,
    )

    daily_food_g = np.round(
        np.clip(
            weight_kg * 14
            + activity_score * 7
            + rng.normal(0, 8, n_rows),
            35,
            180,
        ),
        0,
    ).astype(int)

    vet_visits_year = np.clip(
        rng.poisson(lam=1.0 + age_years / 10, size=n_rows),
        0,
        8,
    )

    adopted = rng.choice(["Sí", "No"], size=n_rows, p=[0.72, 0.28])

    sterilized_probability = np.clip(
        0.45 + age_years * 0.045,
        0.50,
        0.96,
    )
    sterilized = np.where(
        rng.random(n_rows) < sterilized_probability,
        "Sí",
        "No",
    )

    health_score = np.round(
        np.clip(
            9.2
            - age_years * 0.12
            - np.abs(weight_kg - base_weights) * 0.22
            + rng.normal(0, 0.65, n_rows),
            1,
            10,
        ),
        1,
    )

    return pd.DataFrame(
        {
            "id_gato": [f"GATO-{i:04d}" for i in range(1, n_rows + 1)],
            "raza": breeds,
            "sexo": sexes,
            "color": colors,
            "edad_anios": age_years,
            "peso_kg": weight_kg,
            "nivel_actividad": activity,
            "horas_sueno": sleep_hours,
            "alimento_diario_g": daily_food_g,
            "visitas_veterinario_anio": vet_visits_year,
            "adoptado": adopted,
            "esterilizado": sterilized,
            "puntaje_salud": health_score,
        }
    )


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Convert a dataframe to downloadable UTF-8 CSV bytes."""
    return dataframe.to_csv(index=False).encode("utf-8")


st.title("🐈 Plataforma interactiva de datos sintéticos de gatos")
st.write(
    "Genera datos sintéticos, explora estadísticas descriptivas, "
    "aplica filtros y visualiza relaciones entre variables."
)

with st.sidebar:
    st.header("Configuración")

    n_rows = st.slider(
        "Número de gatos",
        min_value=50,
        max_value=5000,
        value=500,
        step=50,
    )

    seed = st.number_input(
        "Semilla aleatoria",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
    )

    st.header("Filtros")

    selected_breeds = st.multiselect(
        "Razas",
        options=BREEDS,
        default=BREEDS,
    )

    selected_sexes = st.multiselect(
        "Sexo",
        options=SEXES,
        default=SEXES,
    )

    age_range = st.slider(
        "Rango de edad",
        min_value=0.0,
        max_value=18.0,
        value=(0.0, 18.0),
        step=0.5,
    )


data = generate_cat_data(n_rows=n_rows, seed=int(seed))

filtered_data = data[
    data["raza"].isin(selected_breeds)
    & data["sexo"].isin(selected_sexes)
    & data["edad_anios"].between(age_range[0], age_range[1])
].copy()

if filtered_data.empty:
    st.warning("No hay registros que coincidan con los filtros seleccionados.")
    st.stop()


tab_overview, tab_eda, tab_charts, tab_data = st.tabs(
    ["Resumen", "EDA estadístico", "Visualizaciones", "Datos"]
)


with tab_overview:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Gatos analizados", f"{len(filtered_data):,}")
    col2.metric(
        "Edad promedio",
        f"{filtered_data['edad_anios'].mean():.1f} años",
    )
    col3.metric(
        "Peso promedio",
        f"{filtered_data['peso_kg'].mean():.2f} kg",
    )
    col4.metric(
        "Salud promedio",
        f"{filtered_data['puntaje_salud'].mean():.1f}/10",
    )

    st.subheader("Vista previa")
    st.dataframe(filtered_data.head(20), use_container_width=True)

    st.download_button(
        label="Descargar datos filtrados en CSV",
        data=dataframe_to_csv_bytes(filtered_data),
        file_name="datos_sinteticos_gatos.csv",
        mime="text/csv",
    )


with tab_eda:
    numeric_columns = (
        filtered_data.select_dtypes(include=np.number).columns.tolist()
    )

    st.subheader("Estadísticas descriptivas")

    descriptive_stats = filtered_data[numeric_columns].describe().T
    descriptive_stats["mediana"] = filtered_data[numeric_columns].median()
    descriptive_stats["varianza"] = filtered_data[numeric_columns].var()
    descriptive_stats["asimetria"] = filtered_data[numeric_columns].skew()

    st.dataframe(
        descriptive_stats.round(3),
        use_container_width=True,
    )

    st.subheader("Frecuencias de variables categóricas")

    categorical_column = st.selectbox(
        "Variable categórica",
        options=[
            "raza",
            "sexo",
            "color",
            "nivel_actividad",
            "adoptado",
            "esterilizado",
        ],
    )

    frequency_table = (
        filtered_data[categorical_column]
        .value_counts(dropna=False)
        .rename_axis(categorical_column)
        .reset_index(name="frecuencia")
    )

    frequency_table["porcentaje"] = (
        frequency_table["frecuencia"] / len(filtered_data) * 100
    ).round(2)

    st.dataframe(
        frequency_table,
        use_container_width=True,
    )

    st.subheader("Matriz de correlación")

    correlation = filtered_data[numeric_columns].corr(numeric_only=True)

    correlation_figure = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        title="Correlación entre variables numéricas",
    )

    st.plotly_chart(
        correlation_figure,
        use_container_width=True,
    )


with tab_charts:
    chart_type = st.radio(
        "Tipo de gráfico",
        [
            "Histograma",
            "Dispersión",
            "Caja por categoría",
            "Barras",
        ],
        horizontal=True,
    )

    if chart_type == "Histograma":
        variable = st.selectbox(
            "Variable numérica",
            options=filtered_data.select_dtypes(include=np.number).columns,
        )

        bins = st.slider(
            "Número de intervalos",
            min_value=5,
            max_value=60,
            value=25,
        )

        figure = px.histogram(
            filtered_data,
            x=variable,
            nbins=bins,
            color="sexo",
            marginal="box",
            title=f"Distribución de {variable}",
        )

    elif chart_type == "Dispersión":
        numeric_options = (
            filtered_data.select_dtypes(include=np.number).columns.tolist()
        )

        x_variable = st.selectbox(
            "Variable X",
            numeric_options,
            index=min(1, len(numeric_options) - 1),
        )

        y_variable = st.selectbox(
            "Variable Y",
            numeric_options,
            index=min(2, len(numeric_options) - 1),
        )

        figure = px.scatter(
            filtered_data,
            x=x_variable,
            y=y_variable,
            color="raza",
            size="puntaje_salud",
            hover_data=[
                "id_gato",
                "sexo",
                "nivel_actividad",
            ],
            title=f"{y_variable} frente a {x_variable}",
        )

    elif chart_type == "Caja por categoría":
        numeric_variable = st.selectbox(
            "Variable numérica",
            options=filtered_data.select_dtypes(include=np.number).columns,
        )

        category_variable = st.selectbox(
            "Variable categórica",
            options=[
                "raza",
                "sexo",
                "nivel_actividad",
                "adoptado",
            ],
        )

        figure = px.box(
            filtered_data,
            x=category_variable,
            y=numeric_variable,
            color=category_variable,
            points="outliers",
            title=f"{numeric_variable} por {category_variable}",
        )

    else:
        category_variable = st.selectbox(
            "Variable para contar",
            options=[
                "raza",
                "sexo",
                "color",
                "nivel_actividad",
                "adoptado",
            ],
        )

        counts = (
            filtered_data[category_variable]
            .value_counts()
            .rename_axis(category_variable)
            .reset_index(name="cantidad")
        )

        figure = px.bar(
            counts,
            x=category_variable,
            y="cantidad",
            text_auto=True,
            title=f"Cantidad de gatos por {category_variable}",
        )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )


with tab_data:
    st.subheader("Consulta interactiva")

    search_text = st.text_input(
        "Buscar por ID, raza, color u otro valor",
        placeholder="Ejemplo: Siamés",
    )

    displayed_data = filtered_data.copy()

    if search_text:
        mask = displayed_data.astype(str).apply(
            lambda column: column.str.contains(
                search_text,
                case=False,
                na=False,
            )
        ).any(axis=1)

        displayed_data = displayed_data[mask]

    st.dataframe(
        displayed_data,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Los datos son completamente sintéticos y se generan localmente "
        "cada vez que cambias el tamaño de la muestra o la semilla."
    )
