#método k-distance feito pelo professor, que irá me auxiliar nas questões
#fiz a função de K_distance para radial
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

def plot_k_distance(X, min_pts, title="K-Distance Plot"):
    """Plota o gráfico K-Distance usando sklearn.NearestNeighbors."""
    k = int(min_pts - 1)

    nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean")
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    kth_distances = distances[:, k]
    k_distances_sorted = np.sort(kth_distances)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(k_distances_sorted)), k_distances_sorted, linewidth=2, label=f'{k}-distance')
    plt.xlabel("Pontos ordenados por distância")
    plt.ylabel(f"{k}-distance")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_k_distance_radial(matriz, min_pts, title="K-Distance"):

    k = int(min_pts - 1)
    ordem= np.sort(matriz, axis=1)
    lugar= ordem[:, k]

    k_distances = np.sort(lugar)

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(k_distances)), k_distances, linewidth=2)
    plt.xlabel("Pontos ordenados por distância")
    plt.ylabel(f"Distância Radial ao {k}º vizinho")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()