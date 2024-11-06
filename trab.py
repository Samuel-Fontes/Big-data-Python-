import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


url = "https://raw.githubusercontent.com/Samuel-Fontes/Big-data-Python-/refs/heads/main/Base%20DATA%20.csv"


df = pd.read_csv(url, delimiter=';')

if 'regiao' not in df.columns:
    df = df.rename(columns={'estado': 'regiao'})

print("DataFrame com os dados de utilização do SUS e planos de saúde:\n", df)


plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(df['regiao']))

plt.bar(index, df['sus_dep'], bar_width, label='Dependente do SUS  (%)')
plt.bar(index + bar_width, df['sus_ndep'], bar_width, label='Plano de Saúde (%)')

plt.xlabel('Região')
plt.ylabel('Porcentagem (%)')
plt.title('Comparação da Dependência do SUS e Plano de Saúde por Região')
plt.xticks(index + bar_width / 2, df['regiao'])
plt.legend()
plt.tight_layout()
plt.show()


X_kmeans = df[['sus_dep', 'sus_ndep']].values

inertia = []
K = range(1, min(10, len(X_kmeans) + 1))
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_kmeans)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'o-')
plt.title("Método do Cotovelo - K-Means")
plt.xlabel("Número de Clusters")
plt.ylabel("Inércia")
plt.xticks(K)
plt.grid()
plt.show()


X_reg = df[['sus_ndep']].values
y_reg = df['sus_dep'].values

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_reg = linear_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_reg)
print(f"Erro Quadrático Médio (Regressão Linear): {mse:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_reg, color='blue', alpha=0.7)
plt.xlabel("Valores Reais (%) - SUS Dependente")
plt.ylabel("Valores Previsto (%) - SUS Dependente")
plt.title("Regressão Linear: Valores Reais vs Valores Previsto")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle='--')
plt.grid()
plt.show()
