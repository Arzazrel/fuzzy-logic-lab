"""
Motore di inferenza fuzzy Mamdani implementato da zero.

Pipeline (strategia FITA - First Infer Then Aggregate):
  1. Fuzzification degli input (valore crisp -> gradi di appartenenza).
  2. Firing strength di ogni regola (t-norm sugli antecedenti).
  3. Implicazione: la firing strength "taglia" (min) il fuzzy set conseguente.
  4. Aggregazione: t-conorm (max) sui conseguenti tagliati.
  5. Defuzzification: fuzzy set aggregato -> valore crisp.

Questa implementazione e' volutamente esplicita e commentata: serve a
mostrare la meccanica dell'inferenza, non a essere la piu' veloce.
Per l'uso pratico esiste la versione con scikit-fuzzy (vedi mamdani_skfuzzy.py).
"""
from __future__ import annotations              # permette type hinting con sintassi moderna
from dataclasses import dataclass, field        # import per creazione di classi 
from typing import Callable                     # import gestione vettori numerici
import numpy as np                              # import funzioni di t-nom , t-conorm, ...

from operators import t_min, s_max, DEFUZZIFIERS


# creazione della classe per la gestione della variabilelinguistica: un universo del discorso e un insieme di termini (etichette) ciascuno con la propria membership function.
@dataclass
class FuzzyVariable:
    name: str                                                   # nome della variabilelinguistica (es. temperatura)
    universe: np.ndarray                                        # array dei valori, vettore NumPy con i valori del dominio discreto
    terms: dict[str, np.ndarray] = field(default_factory=dict)  # dizionario che associa ad ogni etichetta linguistica l'array dei valori della funzione di appartenenza ($\mu$) calcolati sull'universo.

    # aggiunge un termine/etichetta al dizionario 
    def add_term(self, label: str, mf_values: np.ndarray) -> "FuzzyVariable":
        self.terms[label] = mf_values
        return self

    # Grado di appartenenza di `value` al termine `label` (interpolazione lineare sull'universo discreto).
    def membership(self, label: str, value: float) -> float:
        return float(np.interp(value, self.universe, self.terms[label]))


"""Regola: SE (var1 e' term1) AND/OR (var2 e' term2) ALLORA (out e' term).
    `antecedents` e' una lista di coppie (nome_variabile, etichetta).
    `connective` e' 'and' o 'or'.
"""
@dataclass
class Rule:
    antecedents: list[tuple[str, str]]  # lista di coppie della premessa
    consequent: tuple[str, str]         # coppia per  conclusione
    connective: str = "and"             # operatore che lega gli antecedenti (and o or)


class MamdaniFIS:
    # inizializzazione
    def __init__(self, and_op=t_min, or_op=s_max, defuzz="centroid"):
        self.inputs: dict[str, FuzzyVariable] = {}
        self.output: FuzzyVariable | None = None
        self.rules: list[Rule] = []
        self.and_op = and_op
        self.or_op = or_op
        self.defuzz: Callable = DEFUZZIFIERS[defuzz]

    # --- costruzione del sistema -------------------------------------------
    def add_input(self, var: FuzzyVariable):
        self.inputs[var.name] = var
        return self

    def set_output(self, var: FuzzyVariable):
        self.output = var
        return self

    def add_rule(self, rule: Rule):
        self.rules.append(rule)
        return self

    # --- inferenza ----------------------------------------------------------
    # fuzzificazione degli input: per ciascuna coppia (variabile, etchetta) presente negli antecedenti della regola, recupera il grado di appartenenza del valore di input corrente.
    def _firing_strength(self, rule: Rule, crisp_inputs: dict[str, float]) -> float:
        degrees = [
            self.inputs[var].membership(label, crisp_inputs[var])
            for var, label in rule.antecedents
        ]
        if rule.connective == "and":
            result = degrees[0]
            for d in degrees[1:]:
                result = self.and_op(result, d)
        else:  # or
            result = degrees[0]
            for d in degrees[1:]:
                result = self.or_op(result, d)
        return float(result)

    """
    Esegue l'inferenza completa e ritorna il valore crisp di output. 
    Se return_aggregate=True ritorna anche il fuzzy set aggregato,utile per la visualizzazione.
    """
    def infer(self, crisp_inputs: dict[str, float], return_aggregate=False):
        
        assert self.output is not None, "Output non definito"
        aggregated = np.zeros_like(self.output.universe, dtype=float)

        for rule in self.rules:
            alpha = self._firing_strength(rule, crisp_inputs)
            _, out_label = rule.consequent
            consequent_mf = self.output.terms[out_label]
            # implicazione Mamdani: min tra alpha e la MF conseguente
            clipped = np.minimum(alpha, consequent_mf)
            # aggregazione: max (t-conorm)
            aggregated = self.or_op(aggregated, clipped)

        crisp_output = self.defuzz(self.output.universe, aggregated)
        if return_aggregate:
            return crisp_output, aggregated
        return crisp_output
