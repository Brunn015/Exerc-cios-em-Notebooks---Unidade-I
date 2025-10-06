#Os dados das 3 esferas concêntricas:

from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

def generate_concentric_spheres(radii=[3, 15], n_samples_per_sphere=1000, noise=0.2, random_state=42):
    """
    Gera pontos em 3 esferas concêntricas no espaço 3D.
    - radii: lista com os raios das esferas
    - n_samples_per_sphere: pontos em cada esfera
    - noise: variação radial para "espessura" da casca
    """
    rng = np.random.default_rng(random_state)
    X, y = [], []
    
    for i, r in enumerate(radii):
        phi = rng.uniform(0, 2*np.pi, n_samples_per_sphere)       # ângulo azimutal
        costheta = rng.uniform(-1, 1, n_samples_per_sphere)       # cos(theta)
        theta = np.arccos(costheta)                               # ângulo polar
        
        # raio com ruído
        rr = r + noise * rng.standard_normal(n_samples_per_sphere)
        
        # coordenadas cartesianas
        x = rr * np.sin(theta) * np.cos(phi)
        y_ = rr * np.sin(theta) * np.sin(phi)
        z = rr * np.cos(theta)
        
        X.append(np.vstack((x, y_, z)).T)
        y.append(np.full(n_samples_per_sphere, i))
    
    X = np.vstack(X)
    y = np.concatenate(y)
    return X, y

X_spheres, y_spheres = generate_concentric_spheres(radii=[3, 8, 12], n_samples_per_sphere=200, noise=0.4)
scaler = StandardScaler()
X_spheres = scaler.fit_transform(X_spheres)

