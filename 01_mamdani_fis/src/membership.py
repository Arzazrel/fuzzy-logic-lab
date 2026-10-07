"""
Funzioni di appartenenza (membership functions) implementate da zero in NumPy.

Ogni funzione mappa un array di valori dell'universo del discorso `x`
nel corrispondente grado di appartenenza in [0, 1].

Riferimento teorico: le MF piu' usate sono triangolare, trapezoidale e
gaussiana (bell-shaped), come da materiale del corso.
"""
from __future__ import annotations
import numpy as np

"""
Membership function triangolare. Parametri:
    x   : universo del discorso
    abc : (a, b, c) con a <= b <= c. `b` e' il vertice (grado 1), `a` e `c` sono i piedi (grado 0).
"""
def trimf(x: np.ndarray, abc: tuple[float, float, float]) -> np.ndarray:
    
    a, b, c = abc
    assert a <= b <= c, "Richiesto a <= b <= c"
    y = np.zeros_like(x, dtype=float)
    # ramo crescente a < x <= b
    if b > a:
        idx = (x > a) & (x <= b)
        y[idx] = (x[idx] - a) / (b - a)
    # ramo decrescente b < x < c
    if c > b:
        idx = (x > b) & (x < c)
        y[idx] = (c - x[idx]) / (c - b)
    # vertice
    y[x == b] = 1.0
    return y

# Membership function trapezoidale con parametri (a, b, c, d).
def trapmf(x: np.ndarray, abcd: tuple[float, float, float, float]) -> np.ndarray:
    a, b, c, d = abcd
    assert a <= b <= c <= d, "Richiesto a <= b <= c <= d"
    y = np.zeros_like(x, dtype=float)
    if b > a:
        idx = (x > a) & (x < b)
        y[idx] = (x[idx] - a) / (b - a)
    y[(x >= b) & (x <= c)] = 1.0
    if d > c:
        idx = (x > c) & (x < d)
        y[idx] = (d - x[idx]) / (d - c)
    return y

# Membership function gaussiana. `sigma` (spread) e' l'inverso della selettivita': piccolo => funzione stretta e selettiva.
def gaussmf(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    return np.exp(-((x - mean) ** 2) / (2.0 * sigma ** 2))
