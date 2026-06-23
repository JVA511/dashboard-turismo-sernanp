# LIBRERIAS

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# CONFIG

st.set_page_config(
    page_title="Dashboard Turismo SERNANP",
    layout="wide"
)

# CARGA

df = pd.read_csv("SERNANP_Turismo_500_modelo.csv")
modelo = joblib.load("modelo_knn.pkl")

# TITULO

st.title("Dashboard de Visitas Turísticas a Áreas Naturales Protegidas")
st.write("Herramienta de análisis descriptivo y predictivo usando Machine Learning.")

# TABS

tab1, tab2 = st.tabs(["Panel A - Análisis de Datos", "Panel B - Predicción"])

# =========================
# PANEL A
# =========================

with tab1:
    st.header("Panel A - Análisis de Datos")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total de registros", len(df))

    with col2:
        st.metric("Cantidad de ANP", df["ANP"].nunique())

    with col3:
        clase_principal = df["TIPO_ESTADIA_PREDOMINANTE"].mode()[0]
        st.metric("Clase más frecuente", clase_principal)

    st.subheader("Filtros")

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        filtro_procedencia = st.multiselect(
            "Procedencia",
            options=sorted(df["PROCEDENCIA"].dropna().unique()),
            default=sorted(df["PROCEDENCIA"].dropna().unique())
        )

    with col_f2:
        filtro_edad = st.multiselect(
            "Edad",
            options=sorted(df["EDAD"].dropna().unique()),
            default=sorted(df["EDAD"].dropna().unique())
        )

    with col_f3:
        filtro_anio = st.multiselect(
            "Año",
            options=sorted(df["ANIO"].dropna().unique()),
            default=sorted(df["ANIO"].dropna().unique())
        )

    df_filtrado = df[
        (df["PROCEDENCIA"].isin(filtro_procedencia)) &
        (df["EDAD"].isin(filtro_edad)) &
        (df["ANIO"].isin(filtro_anio))
    ]

    st.write("Registros filtrados:", len(df_filtrado))

    st.subheader("Visualizaciones")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.write("Distribución del tipo de estadía")
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        df_filtrado["TIPO_ESTADIA_PREDOMINANTE"].value_counts().plot(kind="bar", ax=ax1)
        ax1.set_title("Tipo de estadía predominante")
        ax1.set_xlabel("Tipo de estadía")
        ax1.set_ylabel("Cantidad")
        ax1.tick_params(axis="x", rotation=0)
        plt.tight_layout()
        st.pyplot(fig1, use_container_width=True)

    with col_g2:
        st.write("Distribución por procedencia")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        df_filtrado["PROCEDENCIA"].value_counts().plot(kind="bar", ax=ax2)
        ax2.set_title("Visitantes por procedencia")
        ax2.set_xlabel("Procedencia")
        ax2.set_ylabel("Cantidad")
        ax2.tick_params(axis="x", rotation=0)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    st.write("Top 10 Áreas Naturales Protegidas con más registros")

    top_anp = df_filtrado["ANP"].value_counts().head(10).sort_values()

    fig3, ax3 = plt.subplots(figsize=(6.5, 3))
    top_anp.plot(kind="barh", ax=ax3)

    ax3.set_title("Top 10 Áreas Naturales Protegidas", fontsize=11)
    ax3.set_xlabel("Cantidad de registros", fontsize=9)
    ax3.set_ylabel("Área Natural Protegida", fontsize=9)
    ax3.tick_params(axis="x", labelsize=8)
    ax3.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)

    st.subheader("Vista de datos")
    st.dataframe(df_filtrado.head(20))

# =========================
# PANEL B
# =========================

with tab2:
    st.header("Panel B - Análisis Predictivo")
    st.write("Ingrese los datos del visitante para predecir el tipo de estadía predominante.")

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        anp = st.selectbox("Área Natural Protegida", sorted(df["ANP"].dropna().unique()))
        sector = st.selectbox("Sector", sorted(df["SECTOR"].dropna().unique()))
        departamento = st.selectbox("Departamento", sorted(df["DEPARTAMENTO"].dropna().unique()))
        provincia = st.selectbox("Provincia", sorted(df["PROVINCIA"].dropna().unique()))
        distrito = st.selectbox("Distrito", sorted(df["DISTRITO"].dropna().unique()))

    with col_p2:
        procedencia = st.selectbox("Procedencia", sorted(df["PROCEDENCIA"].dropna().unique()))
        edad = st.selectbox("Edad", sorted(df["EDAD"].dropna().unique()))
        ubigeo = st.number_input("UBIGEO", min_value=0, value=0)
        anio = st.selectbox("Año", sorted(df["ANIO"].dropna().unique()))
        mes = st.slider("Mes", min_value=1, max_value=12, value=1)

    entrada = pd.DataFrame([{
        "ANP": anp,
        "SECTOR": sector,
        "DEPARTAMENTO": departamento,
        "PROVINCIA": provincia,
        "DISTRITO": distrito,
        "UBIGEO": ubigeo,
        "PROCEDENCIA": procedencia,
        "EDAD": edad,
        "ANIO": anio,
        "MES": mes
    }])

    st.subheader("Datos ingresados")
    st.dataframe(entrada)

    if st.button("Predecir tipo de estadía"):
        prediccion = modelo.predict(entrada)[0]
        probabilidades = modelo.predict_proba(entrada)[0]
        clases = modelo.classes_

        st.success(f"Tipo de estadía predicho: {prediccion}")

        df_prob = pd.DataFrame({
            "Clase": clases,
            "Probabilidad": probabilidades
        })

        df_prob["Probabilidad"] = (df_prob["Probabilidad"] * 100).round(2)

        st.subheader("Probabilidad por clase")
        st.dataframe(df_prob)

        st.write(
            "Esta predicción indica el tipo de estadía que el modelo considera más probable "
            "según los datos ingresados del visitante y del área natural protegida."
        )