"""
### Exercício 3: Detecção de Anomalias com DBSCAN e DTW

O **DTW (Dynamic Time Warping)** mede a similaridade entre séries temporais mesmo quando estão defasadas ou
com velocidades diferentes, alinhando-as de forma elástica. Isso permite detectar padrões semelhantes sem
que a defasagem atrapalhe. 

Tarefas:

Use o dataset de senóides com variação e anomalias simuladas.
Adicione a métrica DTW no DBSCAN.
Experimente diferentes valores de eps e min_samples até que o modelo consiga separar bem séries normais das anômalas.
Plote todas as séries, usando uma cor para as normais e outra para as anomalias detectadas (label = -1).
"""
import matplotlib.pyplot as plt
import numpy as np
from dtaidistance import dtw
import dataset_senoide as data
import dbscan as dbs
import distance as d

def calcular_matriz_dtw(X):
    n = len(X)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = dtw.distance(X[i].astype(np.double), X[j].astype(np.double))
            D[i, j] = D[j, i] = dist
    return D

X_series, y_series = data.generate_time_series_dataset()

print("Calculando a matriz de distâncias DTW,pode levar alguns segundos...")
dtw_matriz = calcular_matriz_dtw(X_series)
print("Cálculo concluído!")

min_pts = 4
eps=1.2
d.plot_k_distance_radial(dtw_matriz, min_pts, title="K-Distance ")

dbscan = dbs.DBSCAN(eps, min_pts, metric='precomputed')
labels = dbscan.fit_predict(dtw_matriz)


plt.figure(figsize=(12, 6))
n_anomalies = 0
for i, series in enumerate(X_series):
    if labels[i] == -1:
        plt.plot(series, color='red', alpha=0.9, label="Anomalia" if n_anomalies == 0 else "")
        n_anomalies += 1
    else:
        plt.plot(series, color='blue', alpha=0.4)

plt.title(f"Detecção de Anomalias com DBSCAN+DTW")
if n_anomalies > 0:
    plt.legend()
plt.show()

print(f"Número de anomalias detectadas: {n_anomalies}")
print(f"Número de clusters normais encontrados: {dbscan.n_clusters_}")
