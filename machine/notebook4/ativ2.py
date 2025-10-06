"""
### Exercício 2: DBSCAN com distância radial

Usando os dados das **3 esferas concêntricas** do exercício anterior:

1. Implemente a **distância radial** e use-a no DBSCAN. A **distância radial** entre dois pontos \(x_i\) e \(x_j\) é a diferença absoluta entre suas distâncias à origem: $d_{\text{radial}}(x_i, x_j) = \big|\;\|x_i\|_2 - \|x_j\|_2\;\big|$
2. Plote o **K-Distance radial** para sugerir `eps`.  
3. Teste combinações de `eps` e `min_samples`.  
4. Visualize em 3D os clusters obtidos e compare com o resultado usando distância euclidiana.  
5. Comente brevemente qual configuração foi melhor e por quê a métrica radial ajuda nesse dataset.
"""
import dados as d
import plotly.express as px
import numpy as np
import distance as k
import dbscan as dbs



X_spheres, y_spheres = d.generate_concentric_spheres(radii=[3, 8, 12], n_samples_per_sphere=200, noise=0.4)

#K-distance e método

min_pts=6
eps=0.5
x = dbs.DBSCAN(metric='radial')
radial_distance = x._calculate_distance_matrix(X_spheres)
modelo = dbs.DBSCAN(eps, min_pts, metric='radial')
distancia=k.plot_k_distance_radial(radial_distance, min_pts, title="K-Distance")

predicted_labels = modelo.fit_predict(X_spheres)


fig = px.scatter_3d(
    x=X_spheres[:, 0],
    y=X_spheres[:, 1],
    z=X_spheres[:, 2],
    color=predicted_labels,
    title=(f"DBSCAN com Distância Radial (eps={eps}, min_pts={min_pts})")
)
fig.update_traces(marker=dict(size=3, opacity=0.8))
fig.show()

n_clusters = len(np.unique(predicted_labels[predicted_labels != -1]))
print(f"Parâmetros: eps={eps}, min_pts={min_pts}")
print(f"Número de clusters encontrados: {n_clusters}")

"""
Essa configuração foi bem superior, após eu coloca para radial,
foi mt mais fácil visualizar qual seria o cotovelo do k-distance, 
e além disso a presença de -1 diminuiu bastante
"""