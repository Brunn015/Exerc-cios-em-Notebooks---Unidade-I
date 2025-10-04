import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from scipy.stats import mode
import kmeans as km


plt.style.use('seaborn-v0_8-whitegrid')

wine = load_wine()
X_data = wine.data[:, [12, 1]] 
y_true = wine.target


K = 3
kmeans_model = km.KMeans(n_clusters=K, random_state=42)
kmeans_model.fit(X_data)

predicted_labels = kmeans_model.labels
centroids = kmeans_model.centroids

mapped_labels = np.zeros_like(predicted_labels)

for i in range(K):
    
    mask = (predicted_labels == i)

    labels_do_cluster = y_true[mask]

    if len(labels_do_cluster) > 0:
        mapeamento_real = mode(labels_do_cluster)[0]

        mapped_labels[mask] = mapeamento_real

acuracia = np.mean(mapped_labels == y_true)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

axes[0].scatter(X_data[:, 0], X_data[:, 1], c=predicted_labels, s=50, cmap='viridis', edgecolor='k')
axes[0].scatter(centroids[:, 0], centroids[:, 1], s=250, marker='*', c='red', edgecolor='black', label='Centróides')
axes[0].set_title(f'Clusters Encontrados (Acurácia: {acuracia:.2%})')
axes[0].set_xlabel(wine.feature_names[12])
axes[0].set_ylabel(wine.feature_names[1])
axes[0].legend()
axes[0].grid(True)

scatter = axes[1].scatter(X_data[:, 0], X_data[:, 1], c=y_true, cmap='viridis', s=50, edgecolor='k')
axes[1].set_title('Classes Reais (Gabarito)')
axes[1].set_xlabel(wine.feature_names[12])
axes[1].set_ylabel(wine.feature_names[1])
axes[1].legend(handles=scatter.legend_elements()[0], labels=['Uva 0', 'Uva 1', 'Uva 2'])
axes[1].grid(True)

plt.suptitle('Comparação: K-Means vs. Rótulos Reais', fontsize=16)
plt.show()
