# ==========================================================
# PASSOS MÁGICOS
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, roc_auc_score

# ==========================================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================================

st.set_page_config(
    page_title="Passos Mágicos - Inteligência Social",
    layout="wide",
    page_icon="📊"
)

# ==========================================================
# VISUAL
# ==========================================================

st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stMetric {background-color: white; padding: 15px; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

# ==========================================================
# CONFIGURAÇÕES GLOBAIS
# ==========================================================

FILE_PATH = "https://raw.githubusercontent.com/caioz/POSTECH_Datathon_FASE5/main/BASE%20DE%20DADOS%20PEDE%202024%20-%20DATATHON.xlsx"
INDICADORES = ['IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN']

# ==========================================================
# CACHE
# ==========================================================

@st.cache_data
def carregar_dados():
    df22 = pd.read_excel(FILE_PATH, sheet_name='PEDE2022')
    df23 = pd.read_excel(FILE_PATH, sheet_name='PEDE2023')
    df24 = pd.read_excel(FILE_PATH, sheet_name='PEDE2024')
    return df22, df23, df24


def converter_numerico(df):
    df = df.copy()
    for col in df.columns:
        if any(ind in col for ind in INDICADORES) or "INDE" in col:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ==========================================================
# CARREGAMENTO
# ==========================================================

df22, df23, df24 = carregar_dados()

df22 = converter_numerico(df22)
df23 = converter_numerico(df23)
df24 = converter_numerico(df24)

st.title("📊 Dashboard Executivo - Inteligência Social")
st.caption("Sistema Estratégico de Apoio à Tomada de Decisão")

# ==========================================================
# ABAS
# ==========================================================

aba1, aba2, aba3, aba4 = st.tabs([
    "📊 Visão Estratégica",
    "🤖 Previsão INDE",
    "🎓 Recomendação de Bolsa",
    "⚠️ Risco de Defasagem"
])

# ==========================================================
# 1️⃣ VISÃO ESTRATÉGICA
# ==========================================================

with aba1:

    st.subheader("Indicadores Estratégicos")

    # Identifica colunas INDE
    col_inde_22 = [c for c in df22.columns if "INDE" in c][0]
    col_inde_23 = [c for c in df23.columns if "INDE" in c][0]
    col_inde_24 = [c for c in df24.columns if "INDE" in c][0]
    
    # Calcula médias
    media22 = df22[col_inde_22].mean()
    media23 = df23[col_inde_23].mean()
    media24 = df24[col_inde_24].mean()
    
    # Crescimento
    crescimento = ((media24 - media22) / media22) * 100 if media22 != 0 else 0

    col1, col2, col3 = st.columns(3)

    col1.metric("INDE Médio 2024", round(media24,2))
    col2.metric("Crescimento 3 anos", f"{crescimento:.1f}%")
    col3.metric("Total Alunos 2024", len(df24))

    st.subheader("Evolução do INDE")

    fig, ax = plt.subplots()
    
    anos = ["2022","2023","2024"]
    valores = [media22, media23, media24]
    
    ax.plot(anos, valores, marker="o", linewidth=3)
    ax.set_ylim(0,10)
    ax.grid(True)
    
    # Adiciona rótulos
    for i, v in enumerate(valores):
        ax.text(i, v + 0.2, f"{v:.2f}", ha='center', fontweight='bold')
    
    st.pyplot(fig)

# ==========================================================
# 2️⃣ PREVISÃO INDE
# ==========================================================

with aba2:

    st.subheader("Modelo de Previsão de Desempenho")

    features = ['RA'] + [c for c in df23.columns if any(ind == c[:3] for ind in INDICADORES)]

    df_merge = pd.merge(
        df23[features],
        df24[['RA','INDE 2024']],
        on="RA"
    ).dropna()

    X = df_merge.drop(columns=["RA","INDE 2024"])
    y = df_merge["INDE 2024"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    col1, col2 = st.columns(2)
    col1.metric("R²", round(r2_score(y_test,preds),3))
    col2.metric("MAE", round(mean_absolute_error(y_test,preds),3))

    st.subheader("Importância dos Indicadores")

    importances = pd.Series(model.feature_importances_, index=X.columns)
    fig, ax = plt.subplots()
    importances.sort_values().plot(kind="barh", ax=ax)
    st.pyplot(fig)

# ==========================================================
# 3️⃣ RECOMENDAÇÃO DE BOLSA
# ==========================================================

with aba3:

    st.subheader("Modelo de Recomendação Estratégica")

    col_inde = [c for c in df23.columns if "INDE" in c][0]
    col_pedra = [c for c in df23.columns if "Pedra" in c][0]

    df23["Candidato_Bolsa"] = np.where(
        (df23[col_inde] >= 8.5) &
        (df23[col_pedra].isin(["Ametista","Topázio"])),
        1,0
    )

    cols_X = [c for c in df23.columns if any(ind == c[:3] for ind in INDICADORES)]
    df_model = df23.dropna(subset=cols_X + ["Candidato_Bolsa"])

    X = df_model[cols_X]
    y = df_model["Candidato_Bolsa"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    clf = RandomForestClassifier(class_weight="balanced",random_state=42)
    clf.fit(X_train,y_train)

    probas = clf.predict_proba(X_test)[:,1]

    col1, col2 = st.columns(2)
    col1.metric("ROC-AUC", round(roc_auc_score(y_test,probas),3))
    col2.metric("Total Candidatos Elite", int(y.sum()))

    st.write(pd.DataFrame(classification_report(y_test, clf.predict(X_test), output_dict=True)).transpose())

# ==========================================================
# 4️⃣ RISCO DE DEFASAGEM
# ==========================================================

with aba4:

    st.subheader("Modelo de Risco de Defasagem")

    df24["Risco_Defasagem"] = np.where(
        df24["IAN"] < 10,
        1,0
    )

    df_risco = pd.merge(
        df23[['RA','IEG','IDA','IPS','IPV']],
        df24[['RA','Risco_Defasagem']],
        on="RA"
    ).dropna()

    X = df_risco[['IEG','IDA','IPS','IPV']]
    y = df_risco['Risco_Defasagem']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    model_risco = RandomForestClassifier(class_weight="balanced",random_state=42)
    model_risco.fit(X_train,y_train)

    prob_risco = model_risco.predict_proba(X_test)[:,1]

    col1, col2 = st.columns(2)
    col1.metric("ROC-AUC Risco", round(roc_auc_score(y_test,prob_risco),3))
    col2.metric("Alunos em Risco 2024", int(df24["Risco_Defasagem"].sum()))

    st.subheader("Importância dos Fatores de Risco")

    importances = pd.Series(model_risco.feature_importances_, index=X.columns)
    fig, ax = plt.subplots()
    importances.sort_values().plot(kind="barh", ax=ax)
    st.pyplot(fig)
