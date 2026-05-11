# ==========================================================
# PROJETO PASSOS MÁGICOS - STREAMLIT APP
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, classification_report


# ==============================
# CONFIGURAÇÕES
# ==============================

FILE_PATH = "https://raw.githubusercontent.com/caioz/POSTECH_Datathon_FASE5/main/BASE%20DE%20DADOS%20PEDE%202024%20-%20DATATHON.xlsx"
INDICADORES = ['IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN']


# ==============================
# CACHE DE DADOS (IMPORTANTÍSSIMO)
# ==============================

@st.cache_data
def carregar_dados():
    df_22 = pd.read_excel(FILE_PATH, sheet_name='PEDE2022')
    df_23 = pd.read_excel(FILE_PATH, sheet_name='PEDE2023')
    df_24 = pd.read_excel(FILE_PATH, sheet_name='PEDE2024')
    return df_22, df_23, df_24


def converter_indicadores_numerico(df):
    df = df.copy()
    for col in df.columns:
        if any(ind in col for ind in INDICADORES) or "INDE" in col:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ==============================
# INTERFACE
# ==============================

st.set_page_config(layout="wide")
st.title("📊 Projeto Passos Mágicos - Inteligência Social")

df_22, df_23, df_24 = carregar_dados()

df_22 = converter_indicadores_numerico(df_22)
df_23 = converter_indicadores_numerico(df_23)
df_24 = converter_indicadores_numerico(df_24)


menu = st.sidebar.selectbox(
    "Selecione a análise:",
    [
        "Diagnóstico Institucional",
        "Previsão INDE 2024",
        "Classificador de Bolsa"
    ]
)

# ==========================================================
# 1️⃣ DIAGNÓSTICO
# ==========================================================

if menu == "Diagnóstico Institucional":

    st.subheader("Evolução do INDE Médio")

    media_22 = df_22.filter(like="INDE").mean().values[0]
    media_23 = df_23.filter(like="INDE").mean().values[0]
    media_24 = df_24.filter(like="INDE").mean().values[0]

    fig, ax = plt.subplots()
    ax.plot(["2022","2023","2024"],
            [media_22, media_23, media_24],
            marker="o")

    ax.set_ylim(0,10)
    st.pyplot(fig)

    st.metric("INDE Médio 2024", round(media_24,2))


# ==========================================================
# 2️⃣ PREVISÃO INDE
# ==========================================================

elif menu == "Previsão INDE 2024":

    st.subheader("Modelo de Previsão de Desempenho")

    features_23 = ['RA'] + [c for c in df_23.columns if any(ind == c[:3] for ind in INDICADORES)]

    df_merge = pd.merge(
        df_23[features_23],
        df_24[['RA','INDE 2024']],
        on="RA"
    ).dropna()

    X = df_merge.drop(columns=["RA","INDE 2024"])
    y = df_merge["INDE 2024"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    col1, col2 = st.columns(2)
    col1.metric("R²", round(r2,3))
    col2.metric("MAE", round(mae,3))

    # Feature Importance
    st.subheader("Importância dos Indicadores")

    importances = pd.Series(model.feature_importances_, index=X.columns)
    fig, ax = plt.subplots()
    importances.sort_values().plot(kind="barh", ax=ax)
    st.pyplot(fig)


# ==========================================================
# 3️⃣ CLASSIFICADOR DE BOLSA
# ==========================================================

elif menu == "Classificador de Bolsa":

    st.subheader("Modelo de Recomendação de Bolsa")

    col_inde = [c for c in df_23.columns if "INDE" in c][0]
    col_pedra = [c for c in df_23.columns if "Pedra" in c][0]

    df_23["Candidato_Bolsa"] = np.where(
        (df_23[col_inde] >= 8.5) &
        (df_23[col_pedra].isin(["Ametista","Topázio"])),
        1, 0
    )

    cols_X = [c for c in df_23.columns if any(ind == c[:3] for ind in INDICADORES)]
    df_model = df_23.dropna(subset=cols_X + ["Candidato_Bolsa"])

    X = df_model[cols_X]
    y = df_model["Candidato_Bolsa"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)

    report = classification_report(y_test, clf.predict(X_test), output_dict=True)

    st.write("### Métricas")
    st.write(pd.DataFrame(report).transpose())