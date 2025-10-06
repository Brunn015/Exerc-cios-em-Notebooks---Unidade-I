"""
Exercício 1: Ajuste de Parâmetros no DBSCAN em 3D
Com os dados das três esferas concêntricas, realize:

Plotar o K-Distance para diferentes valores de min_pts e sugerir um intervalo adequado para eps.
Selecionar os melhores parâmetros de min_pts e eps.
Visualizar em 3D os clusters encontrados (cores diferentes) e comentar a escolha de eps e min_samples.
"""
import dbscan as dbs
import distance as k
import numpy as np
import dados 
import plotly.express as px
from dados import X_spheres, y_spheres

#K-distance
#com a minha analise, eu utilizarei min_pts=6 e eps=0.3
eps=0.3
min_pts=6
k.plot_k_distance(X_spheres, min_pts, title="K-Distance")
metodo=dbs.DBSCAN(eps, min_pts)
predicted_labels=metodo.fit_predict(X_spheres)


#plot 3d
fig = px.scatter_3d(
    x=X_spheres[:, 0],
    y=X_spheres[:, 1],
    z=X_spheres[:, 2],
    color= predicted_labels, 
)
fig.update_traces(marker=dict(size=3))
fig.show()


n_clusters = len(np.unique(predicted_labels[predicted_labels != -1]))
print(f"Parâmetros: eps={eps}, min_pts={min_pts}")
print(f"Número de clusters encontrados: {n_clusters}")