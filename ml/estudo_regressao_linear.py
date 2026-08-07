import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import xy

df = pd.read_csv("dados_previsao_custo_entrega.csv")

# Verificar correlação
print(df.corr())

df_otimizado = df[['distancia_km', 'peso_kg', 'indice_combustivel', 'custo_entrega']]

X = df_otimizado[['distancia_km', 'peso_kg', 'indice_combustivel']]
y = df_otimizado['custo_entrega']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

compare = pd.DataFrame({
    'Real': y_test, 
    'Previsto': y_pred,
    'Diferenca_Absoluta': abs(y_test - y_pred) # Opcional: já calcula o erro residual por linha
})

print(model.coef_)
print(compare)

for i, column in enumerate(df_otimizado.columns):
    if i < len(model.coef_):
        print(f"Para cada 1 unidade em {column}: Muda {model.coef_[i]} no custo de entrega")

df_dados_producao = pd.read_csv("dados_producao_previsao.csv")


df_dados_producao['custo_estimado'] = model.predict(df_dados_producao[['distancia_km', 'peso_kg', 'indice_combustivel']])
df_dados_producao.to_csv("Producao_prev.csv")