"""
Exercício 1: Implementação do Average Linkage

Complete a implementação da nossa classe HierarchicalClustering adicionando o método Average Linkage.
Em seguida, teste todos os três métodos de ligação (single, complete, average) 
no dataset simples (X_simple) e compare os resultados.
"""

import numpy as np
import hierarquia as h

X_simple = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])


linkage_methods = ['single', 'average', 'complete']


for method in linkage_methods:
    clusterizador = h.HierarchicalClustering(linkage=method)
    clusterizador.fit(X_simple)
