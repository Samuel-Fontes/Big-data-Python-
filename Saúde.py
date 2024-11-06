import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.ticker as mticker


url = "https://raw.githubusercontent.com/Samuel-Fontes/Big-data-Python-/main/Base%20DATA%20.csv"


df = pd.read_csv(url, delimiter=';')


if 'regiao' not in df.columns:
    df = df.rename(columns={'estado': 'regiao'})

print("DataFrame com os dados de utilização do SUS e planos de saúde:\n", df)


np.random.seed(0)
df['sem_servicos_medicos'] = np.random.uniform(5, 15, size=len(df)) * df['populacao'] / 100


cores_azul = ['#1f77b4', '#4f83cc', '#6fa3e3', '#93c1f7', '#b0d4fc', '#dae8fc']


plt.figure(figsize=(12, 7))
bars = plt.bar(df['regiao'], df['sem_servicos_medicos'], color=cores_azul[:len(df)], label='População sem Serviços Médicos Privados')


plt.xlabel('Regiões', fontsize=14, fontweight='bold')
plt.ylabel('População sem Serviços Médicos Privados', fontsize=14, fontweight='bold')
plt.title('População Residente sem Acesso a Serviços Médicos Privados por Região', fontsize=16, fontweight='bold')


plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold', color='#333333')
plt.yticks(fontsize=12, color='#333333')


for bar, value in zip(bars, df['sem_servicos_medicos']):
    plt.text(
        bar.get_x() + bar.get_width() / 2, 
        bar.get_height(), 
        f'{value / 1e6:.2f}M', 
        ha='center', 
        va='bottom', 
        fontsize=12, 
        fontweight='bold', 
        color='#333333'
    )

plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

X = df[['populacao']] 
y = df['sem_servicos_medicos']  

linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred = linear_model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(df['populacao'], df['sem_servicos_medicos'], color='blue', label='Dados reais')
plt.plot(df['populacao'], y_pred, color='red', linestyle='--', label='Linha de Regressão Linear')
plt.xlabel('População Total (em milhões)', fontsize=14, fontweight='bold')
plt.ylabel('População sem Serviços Médicos Privados', fontsize=14, fontweight='bold')
plt.title('Regressão Linear: População Total vs População sem Serviços Médicos Privados', fontsize=16, fontweight='bold')


plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x / 1e6)}M'))

plt.legend(fontsize=12)
plt.grid()
plt.tight_layout()
plt.show()
