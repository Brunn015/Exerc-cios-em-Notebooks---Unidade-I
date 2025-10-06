#dataset da senoide que iei utilizar na questão 3

import numpy as np
import matplotlib.pyplot as plt

def generate_time_series_dataset(n_series=50, length=100, noise=0.1, n_outliers=2, random_state=42):
    rng = np.random.default_rng(random_state)
    X, y = [], []
    t = np.linspace(0, 4*np.pi, length)

    # séries normais: senóide com amplitude e frequência ligeiramente diferentes
    for _ in range(n_series):
        amp = rng.uniform(0.8, 1.2)         # amplitude
        freq = rng.uniform(0.9, 1.1)        # frequência
        phase = rng.uniform(0, 0.5*np.pi)   # pequena defasagem
        series = amp * np.sin(freq * t + phase) + noise * rng.normal(size=length)
        X.append(series)
        y.append(0)  # normal

    # outliers: picos ou deslocamentos fortes
    for _ in range(n_outliers):
        amp = rng.uniform(1.5, 2.0)         # amplitude anômala
        freq = rng.uniform(1.2, 1.5)        # frequência anômala
        series = amp * np.sin(freq * t) + noise * rng.normal(size=length)
        if rng.random() < 0.5:
            series[length//2] += 3  # pico
        else:
            series += rng.normal(2.0, 0.5)  # deslocamento
        X.append(series)
        y.append(-1)  # anomalia

    return np.array(X), np.array(y)