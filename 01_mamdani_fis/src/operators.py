"""
Operatori fuzzy: t-norm (AND), t-conorm (OR) e metodi di defuzzification.

Riferimento teorico: t-norm e t-conorm generalizzano intersezione e unione
crisp. I defuzzifier trasformano un fuzzy set aggregato in un valore crisp.
"""
from __future__ import annotations
import numpy as np


# ----------------- t-norm (AND fuzzy) e t-conorm (OR fuzzy) -----------------
def t_min(a, b):
    """t-norm minimo: la piu' usata per l'AND."""
    return np.minimum(a, b)

# t-norm: prodotto (and).
def t_prod(a, b):
    return np.asarray(a) * np.asarray(b)

# t-conorm massimo: la piu' usata per l'OR.
def s_max(a, b):
    return np.maximum(a, b)


def s_probor(a, b):
    """t-conorm somma probabilistica: a + b - a*b."""
    a, b = np.asarray(a), np.asarray(b)
    return a + b - a * b


# ---------------------------------- Defuzzification ----------------------------------
# Center of Area (COA) / centroide. Metodo di riferimento: tiene conto dell'intera forma dell'aggregato.
def centroid(y: np.ndarray, mu: np.ndarray) -> float:
    denom = np.sum(mu)
    if denom == 0:
        return float(np.mean(y))  # fallback: nessuna regola attivata
    return float(np.sum(y * mu) / denom)

# Mean of Maxima (MOM): media dei punti a massima appartenenza.
def mean_of_maxima(y: np.ndarray, mu: np.ndarray) -> float:
    peak = np.max(mu)
    if peak == 0:
        return float(np.mean(y))
    return float(np.mean(y[np.isclose(mu, peak)]))

# Bisector: il punto che divide l'area dell'aggregato in due meta'.
def bisector(y: np.ndarray, mu: np.ndarray) -> float:
    total = np.sum(mu)
    if total == 0:
        return float(np.mean(y))
    cumulative = np.cumsum(mu)
    idx = np.searchsorted(cumulative, total / 2.0)
    idx = min(idx, len(y) - 1)
    return float(y[idx])


DEFUZZIFIERS = {
    "centroid": centroid,
    "mom": mean_of_maxima,
    "bisector": bisector,
}
