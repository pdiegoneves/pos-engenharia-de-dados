# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.15",
#     "matplotlib==3.11.1",
#     "numpy==2.5.1",
#     "pandas==3.0.5",
#     "scikit-learn==1.9.0",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, FunctionTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Pré-processamento de Dados com pandas e scikit-learn (com Pipelines)
    """)
    return


@app.cell(hide_code=True)
def _():
    df_nulos = pd.DataFrame({'col1': [1, 2, np.nan], 'col2': [4, np.nan, 6]})
    return (df_nulos,)


@app.cell
def _(df_nulos):
    df_nulos
    return


@app.cell
def _(df_nulos):
    print("Valoes nulos por coluna:")
    print(df_nulos.isnull().sum())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Tratar valores nulos
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Preencher 'col1' com a média
    """)
    return


@app.cell
def _(df_nulos):
    df_fill = df_nulos.copy()

    media_col1 = df_fill['col1'].mean()
    df_fill['col1'] = df_fill['col1'].fillna(media_col1)
    print("Dataframe após preencher 'col1' com a média (pandas)")
    print(df_fill)

    media_col2 = df_fill['col2'].mean()
    df_fill['col2'] = df_fill['col2'].fillna(media_col2)
    print("Dataframe após preencher 'col2' com a média (pandas)")
    print(df_fill)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Removendo valores nulos
    """)
    return


@app.cell
def _(df_nulos):
    df_linhas_removidas = df_nulos.copy().dropna()
    print("DataFrame após remover linhas com valores nulos:")
    print(df_linhas_removidas)
    return


@app.cell
def _(df_nulos):
    df_colunas_removidas = df_nulos.dropna(axis=1)
    print("DataFrame após remover colunas com valores nulos:")
    print(df_colunas_removidas)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Usando o SimpleInputer (para integrar em Pipelines)
    """)
    return


@app.cell
def _(df_nulos):
    num_imputer = SimpleImputer(strategy='median')
    filled = num_imputer.fit_transform(df_nulos)
    df_imputed = pd.DataFrame(filled, columns=df_nulos.columns)
    print("DataFrame imputado com SimpleImputer (mediana)")
    print(df_imputed)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Codificação de dados categóricos
    """)
    return


@app.cell
def _():
    df_ohe = pd.DataFrame({
        'produtos': ['camiseta', 'calça', 'jaqueta', 'bermuda', 'camiseta'],
        'cor': ['vermelho', 'azul', 'verde', 'azul', 'azul']})
    df_ohe_pd = pd.get_dummies(df_ohe, columns=['cor'], prefix='cat')
    print("Com pandas.geT_dummies():")
    print(df_ohe_pd)
    return (df_ohe,)


@app.cell
def _(df_ohe):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = ohe.fit_transform(df_ohe[['cor']])
    df_ohe_skl = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(['cor']))
    print("\nCom OneHotEncoder (sklearn):")
    print(df_ohe_skl)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Padronização e Normalização de Escalas
    """)
    return


@app.cell
def _():
    data_scaler = np.array([[10, 100], [20, 200], [30, 300]])
    data_scaler
    return (data_scaler,)


@app.cell
def _(data_scaler):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_scaler)
    print("Dados padronizados (média ~0, desvio ~1)")
    print(np.round(scaled_data, 3))
    return


@app.cell
def _(data_scaler):
    minmax = MinMaxScaler()
    normalized_data = minmax.fit_transform(data_scaler)
    print("Dados normalizados (0-1):")
    print(np.round(normalized_data, 3))
    return


@app.cell
def _():
    # Exemplo de DataFrame para divisão
    df_split = pd.DataFrame({'feature1': range(100), 'feature2': range(100, 200), 'target': np.random.randint(0, 2, 100)})

    # Separa X e y
    X = df_split[['feature1', 'feature2']]
    y = df_split['target']

    # Divide 80% para treino e 20% para teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) #random_state=42 - padrão

    print(f"Tamanho do conjunto X_train: {X_train.shape}")
    print(f"Tamanho do conjunto X_test: {X_test. shape}")
    print("Proporção de classes no treino vs. teste:")
    print(pd.Series(y_train) .value_counts(normalize=True) . rename("train"))
    print(pd.Series(y_test).value_counts (normalize=True) .rename("test"))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
