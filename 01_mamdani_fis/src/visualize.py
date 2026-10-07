"""
Visualizzazioni per il sistema turisti:
  1. Le membership function di input e output.
  2. La superficie di controllo (output in funzione dei due input).

Genera i PNG nella cartella figures/.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

from example_tourists import build_system

# percorso di dove mettere i grafici
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)         # creazione della cartella se non esiste

# Traccia in un'unica figura i grafici 2D di tutte le funzioni di appartenenza per gli input e per l'output.
def plot_membership_functions():
    fis = build_system()                    # per istanziare ed estrarre il modello fuzzy Mamdani.
    variables = [
        ("temperature", "Temperatura (C)"),
        ("sunshine", "Sole (%)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (vname, xlabel) in zip(axes[:2], variables):
        var = fis.inputs[vname]
        for label, mf in var.terms.items():
            ax.plot(var.universe, mf, linewidth=2, label=label)
        ax.set_title(f"Input: {vname}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("grado di appartenenza")
        ax.legend()
        ax.grid(alpha=0.3)

    out = fis.output                        # prendo output
    for label, mf in out.terms.items():
        axes[2].plot(out.universe, mf, linewidth=2, label=label)
    axes[2].set_title("Output: tourists")
    axes[2].set_xlabel("Turisti (%)")
    axes[2].set_ylabel("grado di appartenenza")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "membership_functions.png")
    plt.savefig(path, dpi=120)
    print(f"salvato {path}")

# Genera un grafico tridimensionale (superficie di controllo 3D) che mostra il valore di output calcolato dal sistema al variare continuo dei due input.
def plot_control_surface():
    fis = build_system()
    temps = np.arange(0, 41, 2.0)
    suns = np.arange(0, 101, 5.0)
    T, S = np.meshgrid(temps, suns)
    Z = np.zeros_like(T)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            Z[i, j] = fis.infer(
                {"temperature": T[i, j], "sunshine": S[i, j]})

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(T, S, Z, cmap="viridis", edgecolor="none")
    ax.set_xlabel("Temperatura (C)")
    ax.set_ylabel("Sole (%)")
    ax.set_zlabel("Turisti (%)")
    ax.set_title("Superficie di controllo del FIS")
    fig.colorbar(surf, shrink=0.5, aspect=10)
    path = os.path.join(FIG_DIR, "control_surface.png")
    plt.savefig(path, dpi=120)
    print(f"salvato {path}")

"""
Rappresenta l'entry-point dello script: quando il file viene eseguito direttamente da riga di comando, 
lancia in sequenza le due funzioni per generare ed esportare entrambe le immagini PNG.
"""
if __name__ == "__main__":
    plot_membership_functions()
    plot_control_surface()
