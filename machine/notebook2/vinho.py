"""
Exercícios
Neste exercício, você usará a classe KMeans que construímos para explorar um novo dataset e descobrir seus agrupamentos naturais.

O dataset escolhido é o "Wine dataset", que contém a análise química de vinhos cultivados na mesma região da Itália, 
mas derivados de três cultivares (tipos de uva) diferentes. O objetivo é ver se o K-Means consegue agrupar os vinhos de acordo com seu cultivar, 
usando apenas as suas características químicas.


Tarefa 1: Análise e Seleção de Features

Primeiro, carregue o dataset wine e use a função seaborn.pairplot para visualizar as relações entre as features. 
Analise o gráfico e escolha o par de features que você acredita que melhor separa os 3 grupos. 
Plote um gráfico de dispersão apenas com o par selecionado.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
import pandas as pd
import seaborn as sns
import kmeans as km 

plt.style.use('seaborn-v0_8-whitegrid')


wine = load_wine()
x_total = wine.data
y_target = wine.target 


x_data = x_total[:, [12, 1]]

kmeans_model = km.KMeans(n_clusters=3, max_iter=100, random_state=42)

kmeans_model.fit(x_data)


centroids = kmeans_model.centroids
labels = kmeans_model.labels


fig, axes = plt.subplots(1, 2, figsize=(16, 7))

axes[0].scatter(x_data[:, 0], x_data[:, 1], c=labels, s=50, cmap='viridis', edgecolor='k')
axes[0].scatter(centroids[:, 0], centroids[:, 1], s=250, marker='*', c='red', edgecolor='black', label='Centróides')
axes[0].set_title('Clusters Encontrados pelo K-Means')
axes[0].set_xlabel(wine.feature_names[12])
axes[0].set_ylabel(wine.feature_names[1])
axes[0].legend()
axes[0].grid(True)

scatter = axes[1].scatter(x_data[:, 0], x_data[:, 1], c=y_target, cmap='viridis', s=50, edgecolor='k')
axes[1].set_title('Classes Reais (Gabarito)')
axes[1].set_xlabel(wine.feature_names[12])
axes[1].set_ylabel(wine.feature_names[1])
axes[1].legend(handles=scatter.legend_elements()[0], labels=['Uva 0', 'Uva 1', 'Uva 2'])
axes[1].grid(True)

plt.suptitle('Comparação: K-Means vs. Rótulos Reais', fontsize=16)
plt.show()
plt.show()
#um algoritimo que eu utilizei para encontrar um par de features
"""wine_df = pd.DataFrame(x_data, columns=wine.feature_names)
wine_df['uvas'] = pd.Series(y_target).map({0: '', 1: '', 2: ''})

sns.pairplot(wine_df, hue='uvas', palette='viridis', diag_kind='kde')
plt.suptitle('Pair Plot do Dataset wine', y=1.02) """

