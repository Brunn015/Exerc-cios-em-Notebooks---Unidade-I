"""
Tarefa 2: Encontrando o K Ótimo

Aplique o Método do Cotovelo nas duas features que você escolheu. 
Calcule e plote a inércia (WCSS) para K de 1 a 10. Com base no seu gráfico, 
qual parece ser o número ideal de clusters?
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
import pandas as pd
import seaborn as sns
import kmeans as km
from vinho import x_data


k_range = range(1, 11)
inertias = []
clustering_results = []

for k in k_range:
    model = km.KMeans(n_clusters=k, max_iter=150, random_state=42)
    model.fit(x_data)
    
    current_inertia = 0
    for i in range(k):
        cluster_points = x_data[model.labels == i]
        current_inertia += np.sum((cluster_points - model.centroids[i])**2)
    
    inertias.append(current_inertia)
    clustering_results.append({'labels': model.labels, 'centroids': model.centroids})

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inércia (WCSS)')
plt.title('Método do Cotovelo para Encontrar o K Ótimo')
plt.xticks(k_range)
plt.grid(True)
plt.show()

#De acordo com o metodo do cotovelo utilizado, a quantidade mais adequada será 3 clusters