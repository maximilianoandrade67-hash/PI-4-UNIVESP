import pandas as pd
import numpy as np
import inflection
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor

import streamlit as st

# ---------------------------------------------------------
# FUNÇÕES AUXILIARES
# ---------------------------------------------------------

def definir_estacao(mes: int) -> str:
    if 1 <= mes <= 3:
        return "verao"
    elif 4 <= mes <= 6:
        return "outono"
    elif 7 <= mes <= 9:
        return "inverno"
    else:
        return "primavera"


def data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte colunas para tipos numéricos, de acordo com os nomes que vêm dos seus arquivos.
    """

    colunas_float = [
        "presença_de_criadouros",
        "água_parada_em_terrenos_baldios",
        "precipitação",
        "umidade",
        "temperatura",
        "falta_de_coleta_de_lixo",
        "áreas_com_acúmulo_de_entulhos",
        "falta_de_controle_de_pragas",
        "taxa_de_tratamento_de_esgoto",
        "condições_de_moradia_precárias",
        "falta_de_acesso_a_serviços_de_saúde",
        "migração_de_pessoas_de_áreas_endêmicas",
        "transporte_de_mercadorias_em_áreas_urbanas",
        "outros",
    ]

    colunas_int = [
        "presença_de_piscinas_sem_manutenção",
        "presença_de_recipientes_sem_tampas",
        "presença_do_mosquito",
    ]

    for c in colunas_float:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(",", ".")
            df[c] = df[c].astype(float)

    for c in colunas_int:
        if c in df.columns:
            df[c] = df[c].astype(int)

    if "casos_de_dengue" in df.columns:
        df["casos_de_dengue"] = df["casos_de_dengue"].astype(int)

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["estacao_do_ano"] = df["month"].apply(definir_estacao)
    return df


def encoding_cycles(df: pd.DataFrame, cols) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            if "week" in col:
                df[col + "_sin"] = np.sin(df[col] * 2 * np.pi / 52)
                df[col + "_cos"] = np.cos(df[col] * 2 * np.pi / 52)
            if "month" in col:
                df[col + "_sin"] = np.sin(df[col] * 2 * np.pi / 12)
                df[col + "_cos"] = np.cos(df[col] * 2 * np.pi / 12)
            if "day" in col:
                df[col + "_sin"] = np.sin(df[col] * 2 * np.pi / 30)
                df[col + "_cos"] = np.cos(df[col] * 2 * np.pi / 30)
    return df


# ---------------------------------------------------------
# CARREGAMENTO E PREPARO DOS DADOS (USANDO SEUS CSVs)
# ---------------------------------------------------------

def load_data() -> pd.DataFrame:
    # Lê seus arquivos diretamente da pasta do projeto
    df1 = pd.read_csv("agua_parada.csv", encoding="latin1")
    df2 = pd.read_csv("casos_dengue.csv", encoding="latin1")
    df3 = pd.read_csv("condicoes_climaticas.csv", encoding="latin1")
    df4 = pd.read_csv("falta_higiene.csv", encoding="latin1")
    df5 = pd.read_csv("fato.csv", encoding="latin1")

    # Merge pelo campo Date
    df = (
        df1
        .merge(df2, on="Date", how="left")
        .merge(df3, on="Date", how="left")
        .merge(df4, on="Date", how="left")
        .merge(df5, on="Date", how="left")
    )

    # Converter nomes de colunas para snake_case com inflection
    snake = lambda x: inflection.underscore(x)
    df.columns = [snake(c) for c in df.columns]

    # Agora temos: date, casos_de_dengue, precipitação, umidade, temperatura, etc.

    # Garante que existe coluna 'date'
    if "date" not in df.columns:
        raise ValueError(f"Coluna 'date' não encontrada. Colunas: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"])

    # Tratar 'indisponivel' e NaN
    df = df.replace("indisponivel", np.nan)
    df = df.fillna(method="ffill", limit=7)

    # Ajustar tipos numéricos
    df = data_types(df)

    # Features de data
    df = feature_engineering(df)

    # Encoding cíclico de dia/mês/semana
    df = encoding_cycles(df, ["week_of_year", "month", "day"])

    # One-hot de estação do ano
    if "estacao_do_ano" in df.columns:
        df = pd.get_dummies(df, columns=["estacao_do_ano"])

    # Remover colunas já transformadas
    df = df.drop(columns=["month", "day", "week_of_year"], errors="ignore")

    return df


# ---------------------------------------------------------
# SPLIT TREINO/TESTE E TREINO DO MODELO
# ---------------------------------------------------------

def split_train_test(df: pd.DataFrame, train_frac: float = 0.8):
    df_sorted = df.sort_values("date").reset_index(drop=True)
    n = len(df_sorted)
    split_idx = max(1, int(n * train_frac))  # evita split em 0

    train = df_sorted.iloc[:split_idx].copy()
    test = df_sorted.iloc[split_idx:].copy()

    # Se test ficar vazio, forçamos pelo menos 1 linha
    if test.empty and len(train) > 1:
        test = train.tail(1).copy()
        train = train.iloc[:-1].copy()

    return train, test


def train_model(df: pd.DataFrame):
    if "casos_de_dengue" not in df.columns:
        raise ValueError(
            f"Coluna 'casos_de_dengue' não encontrada. Colunas disponíveis: {list(df.columns)}"
        )

    train, test = split_train_test(df, train_frac=0.8)

    y_train = train["casos_de_dengue"]
    y_test = test["casos_de_dengue"]

    # Features numéricas (excluindo target e date)
    features = [
        c
        for c in df.columns
        if c not in ["date", "casos_de_dengue"]
        and df[c].dtype in [np.int64, np.float64, np.int32, np.float32]
    ]

    X_train = train[features]
    X_test = test[features]

    model = ExtraTreesRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return model, features, mae, rmse


# ---------------------------------------------------------
# GRÁFICOS
# ---------------------------------------------------------

def plot_series(df: pd.DataFrame):
    # Série anual
    if "casos_de_dengue" not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Coluna 'casos_de_dengue' não encontrada", ha="center")
        ax.axis("off")
        return fig

    aux = df.groupby("year")["casos_de_dengue"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(aux["year"], aux["casos_de_dengue"], marker="o")
    ax.set_title("Casos de Dengue por Ano")
    ax.set_xlabel("Ano")
    ax.set_ylabel("Casos de dengue")
    ax.grid(True)
    return fig


def plot_corr(df: pd.DataFrame):
    numeric = df.select_dtypes(include=[np.number])

    if numeric.shape[1] == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sem variáveis numéricas para correlação", ha="center")
        ax.axis("off")
        return fig

    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, ax=ax, cmap="coolwarm")
    ax.set_title("Correlação entre Variáveis Numéricas")
    return fig


def plot_importance(model, features):
    importances = model.feature_importances_
    df_imp = (
        pd.DataFrame({"feature": features, "importance": importances})
        .sort_values("importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_imp["feature"], df_imp["importance"])
    ax.set_title("Importância das Variáveis (ExtraTrees)")
    ax.set_xlabel("Importância")
    return fig


# ---------------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------------

def main():
    st.set_page_config(page_title="Dashboard Dengue", layout="wide")

    st.title("📊 Dashboard de Previsão de Casos de Dengue")

    with st.spinner("Carregando e preparando os dados..."):
        df = load_data()

    st.success("Dados carregados com sucesso!")

    # ----------------- EDA -----------------
    st.subheader("📈 Série histórica de casos de dengue")
    fig_series = plot_series(df)
    st.pyplot(fig_series)

    st.subheader("🔥 Correlação entre variáveis numéricas")
    fig_corr = plot_corr(df)
    st.pyplot(fig_corr)

    # ----------------- MODELO -----------------
    st.sidebar.header("Treinamento de Modelo")
    st.sidebar.write("Modelo: ExtraTrees Regressor")

    if st.sidebar.button("Treinar modelo"):
        with st.spinner("Treinando modelo ExtraTrees..."):
            model, features, mae, rmse = train_model(df)

        st.subheader("🎯 Desempenho do Modelo (ExtraTrees)")
        col1, col2 = st.columns(2)
        col1.metric("MAE", f"{mae:,.2f}")
        col2.metric("RMSE", f"{rmse:,.2f}")

        st.subheader("🌳 Importância das variáveis")
        fig_imp = plot_importance(model, features)
        st.pyplot(fig_imp)


if __name__ == "__main__":
    main()
