"""
Stesso sistema turisti, implementato con scikit-fuzzy.

Scopo: mostrare come la libreria di riferimento astrae la pipeline che in
example_tourists.py e' scritta a mano. Confrontando gli output dei due file
si verifica che l'implementazione from-scratch e' corretta.
"""
from __future__ import annotations
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# creazione del sistema
def build_system():
    # definisco gli intervalli delle variabili (val_min, val_max_escluso, intervallo)
    temperature = ctrl.Antecedent(np.arange(0, 41, 1), "temperature")   
    sunshine = ctrl.Antecedent(np.arange(0, 101, 1), "sunshine")
    tourists = ctrl.Consequent(np.arange(0, 101, 1), "tourists")

    # definire i valori per la variabile delle temperatura (triangolari)
    temperature["cold"] = fuzz.trimf(temperature.universe, [0, 0, 20])
    temperature["warm"] = fuzz.trimf(temperature.universe, [10, 20, 30])
    temperature["hot"] = fuzz.trimf(temperature.universe, [20, 40, 40])

    sunshine["cloudy"] = fuzz.trimf(sunshine.universe, [0, 0, 50])
    sunshine["partly_sunny"] = fuzz.trimf(sunshine.universe, [20, 50, 80])
    sunshine["sunny"] = fuzz.trimf(sunshine.universe, [50, 100, 100])

    tourists["low"] = fuzz.trimf(tourists.universe, [0, 0, 50])
    tourists["medium"] = fuzz.trimf(tourists.universe, [0, 50, 100])
    tourists["high"] = fuzz.trimf(tourists.universe, [50, 100, 100])
    
    # creazione delle regole del sistema
    r1 = ctrl.Rule(temperature["hot"] & sunshine["sunny"], tourists["high"])
    r2 = ctrl.Rule(temperature["warm"] & sunshine["partly_sunny"], tourists["medium"])
    r3 = ctrl.Rule(temperature["cold"] | sunshine["cloudy"], tourists["low"])

    system = ctrl.ControlSystem([r1, r2, r3])       # aggiungo le regole al sistema
    return ctrl.ControlSystemSimulation(system)     

# metodo per dare input, calcolare e restitire il risultato
def predict(sim, temp: float, sun: float) -> float:
    sim.input["temperature"] = temp
    sim.input["sunshine"] = sun
    sim.compute()
    return float(sim.output["tourists"])


if __name__ == "__main__":
    sim = build_system()
    for temp, sun in [(19, 60), (35, 90), (5, 20), (25, 55)]:
        y = predict(sim, temp, sun)
        print(f"temp={temp:>3}C  sun={sun:>3}%  ->  turisti stimati = {y:5.1f}%")
