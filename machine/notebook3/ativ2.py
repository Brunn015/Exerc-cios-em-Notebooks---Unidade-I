"""
Exercício 2: Análise do Dataset Wine - Seleção de Features e Comparação de Métodos
Aplique a clusterização hierárquica do SciPy ao dataset Wine. Primeiro, você deve selecionar um bom par de features para visualização bidimensional, depois comparar diferentes métodos de ligação.

Tarefas:

Carregue o dataset Wine e explore suas features
Selecione as duas melhores features para visualização (analise correlações, variâncias, etc.)
Aplique os 4 métodos de ligação ('single', 'complete', 'average', 'ward') usando scipy.cluster.hierarchy.linkage
Crie dendrogramas para cada método
Determine visualmente qual método produz a melhor separação
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

wine = load_wine()
X = wine.data
y = wine.target 

df = pd.DataFrame(X, columns=wine.feature_names)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=wine.feature_names)

features_selecionadas = ['flavanoids', 'proline']
X_selecionado = df_scaled[features_selecionadas].values

metodos_ligacao = ['single', 'complete', 'average', 'ward']

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Comparação de Dendrogramas por Método de Ligação', fontsize=16)


for metodo, ax in zip(metodos_ligacao, axes.flatten()):

    linked = linkage(X_selecionado, method=metodo)

    dendrogram(linked,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=True,
               ax=ax)
    ax.set_title(f'Método: {metodo.capitalize()}')
    ax.set_ylabel('Distância')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#melhor método foi o ward