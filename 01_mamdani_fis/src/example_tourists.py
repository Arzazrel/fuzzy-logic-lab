"""
Esempio: sistema fuzzy di predizione dei turisti (dal materiale del corso).

Input:
  - temperatura [0, 40] gradi C  : cold / warm / hot
  - sunshine    [0, 100] %       : cloudy / partly_sunny / sunny
Output:
  - tourists    [0, 100] %       : low / medium / high

Regole (dalle slide):
  R1: IF temp hot  AND sun sunny         THEN tourists high
  R2: IF temp warm AND sun partly_sunny  THEN tourists medium
  R3: IF temp cold OR  sun cloudy         THEN tourists low
"""
from __future__ import annotations
import numpy as np

from membership import trimf
from mamdani import FuzzyVariable, Rule, MamdaniFIS

# funzione che crea il sistema definito e lo prova con valori prestabiliti e poi mostra i relativi risultati
def build_system(defuzz: str = "centroid") -> MamdaniFIS:
    # --- variabile di input: temperatura ---
    t_universe = np.arange(0, 41, 1.0)                              # range totale dell'universo della variabile
    temperature = FuzzyVariable("temperature", t_universe)          # creo variabile e gli assegno l'universo
    temperature.add_term("cold", trimf(t_universe, (0, 0, 20)))
    temperature.add_term("warm", trimf(t_universe, (10, 20, 30)))
    temperature.add_term("hot", trimf(t_universe, (20, 40, 40)))

    # --- variabile di input: sole ---
    s_universe = np.arange(0, 101, 1.0)
    sunshine = FuzzyVariable("sunshine", s_universe)
    sunshine.add_term("cloudy", trimf(s_universe, (0, 0, 50)))
    sunshine.add_term("partly_sunny", trimf(s_universe, (20, 50, 80)))
    sunshine.add_term("sunny", trimf(s_universe, (50, 100, 100)))

    # --- variabile di output: turisti ---
    o_universe = np.arange(0, 101, 1.0)
    tourists = FuzzyVariable("tourists", o_universe)
    tourists.add_term("low", trimf(o_universe, (0, 0, 50)))
    tourists.add_term("medium", trimf(o_universe, (0, 50, 100)))
    tourists.add_term("high", trimf(o_universe, (50, 100, 100)))

    # definisco input ed output del sistema
    fis = MamdaniFIS(defuzz=defuzz)
    fis.add_input(temperature).add_input(sunshine).set_output(tourists)

    # creazione delle regole per il sistema
    fis.add_rule(Rule([("temperature", "hot"), ("sunshine", "sunny")],
                      ("tourists", "high"), connective="and"))
    fis.add_rule(Rule([("temperature", "warm"), ("sunshine", "partly_sunny")],
                      ("tourists", "medium"), connective="and"))
    fis.add_rule(Rule([("temperature", "cold"), ("sunshine", "cloudy")],
                      ("tourists", "low"), connective="or"))
    return fis


if __name__ == "__main__":
    fis = build_system()
    # esempio dalle slide: temp = 19, sole = 60
    for temp, sun in [(19, 60), (35, 90), (5, 20), (25, 55)]:
        y = fis.infer({"temperature": temp, "sunshine": sun})
        print(f"temp={temp:>3}C  sun={sun:>3}%  ->  turisti stimati = {y:5.1f}%")
