"""
Exercício 3: Determinação do Número Ótimo de Clusters
Com base no melhor método de ligação identificado no Exercício 2, determine o número ótimo de clusters para o dataset Wine usando análise visual do dendrograma e validação com os rótulos verdadeiros.

Tarefas:

Use o melhor método identificado no exercício anterior
Crie um dendrograma detalhado com linha de corte ajustável
Teste diferentes números de clusters (2, 3, 4, 5) usando fcluster
Para cada número de clusters, visualize os clusters no scatter plot
Determine o número ótimo de clusters justificando sua escolha
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

wine = load_wine()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(wine.data)
df_scaled = pd.DataFrame(X_scaled, columns=wine.feature_names)
features_selecionadas = ['flavanoids', 'proline']
X_selecionado = df_scaled[features_selecionadas].values

linked_ward = linkage(X_selecionado, method='ward')


num_clusters_teste = [2, 3, 4, 5]

# Criando uma figura para os subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Visualização da Clusterização com Diferentes k', fontsize=18)

# Iterar sobre os valores de k e plotar
for k, ax in zip(num_clusters_teste, axes.flatten()):
    # Usando fcluster para obter os rótulos
    # 'maxclust' é o critério para formar no máximo 'k' clusters
    labels = fcluster(linked_ward, k, criterion='maxclust')

    # 4. Visualizar os clusters no scatter plot
    ax.scatter(X_selecionado[:, 0], X_selecionado[:, 1], c=labels, cmap='viridis')
    ax.set_title(f'k = {k} clusters')
    ax.set_xlabel('Flavanoids (Padronizado)')
    ax.set_ylabel('Proline (Padronizado)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

plt.figure(figsize=(15, 8))
plt.title('Dendrograma com Método Ward e Linha de Corte', fontsize=16)
dendrogram(linked_ward,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.ylabel('Distância (Ward)')
plt.xlabel('Índice da Amostra de Vinho')

plt.axhline(y=8.5, color='r', linestyle='--')
plt.text(x=1000, y=9, s='Linha de corte para 3 clusters', color='red', va='bottom', ha='center')
plt.show()

#esse ultimo dendograma demonstra a melhor quantidade de clusters(3)