"""
Programma interattivo per il sistema turisti (Mamdani FIS).

Chiede all'utente temperatura e percentuale di sole, verifica che i valori
siano all'interno del rispettivo universo del discorso, ed esegue l'inferenza
stampando il numero di turisti stimato.

I range ammessi vengono letti direttamente dal sistema fuzzy (dagli universi
delle variabili), cosi' il controllo resta corretto anche se in futuro i
range vengono modificati in example_tourists.py.

Uso:
    python interactive.py                 # modalita' interattiva (chiede i valori)
    python interactive.py 19 60           # valori da riga di comando
"""
from __future__ import annotations
from example_tourists import build_system
import sys

# Minimo e massimo dell'universo del discorso di una variabile fuzzy.
def variable_range(var):
    return float(var.universe.min()), float(var.universe.max())

# Stringa descrittiva: range ammesso e termini linguistici disponibili.
def describe_variable(var, label: str, unit: str):
    lo, hi = variable_range(var)                            # prende il min e max
    terms = ", ".join(var.terms.keys())                     # costruisce la stringa da stampare
    return (f"  {label:<12} [{lo:g} .. {hi:g}] {unit}\n"
            f"               termini: {terms}")

# metodo per la generazione della legenda nella UI
def print_header(fis):
    print("=" * 60)
    print("  Sistema fuzzy di predizione turisti (Mamdani FIS)")
    print("=" * 60)
    print("\nValori ammessi per gli input:\n")
    print(describe_variable(fis.inputs["temperature"], "Temperatura", "gradi C"))
    print(describe_variable(fis.inputs["sunshine"], "Sole", "%"))
    lo, hi = variable_range(fis.output)
    print(f"\nOutput:\n  {'Turisti':<12} [{lo:g} .. {hi:g}] %")
    print("-" * 60)

# Chiede un valore finche' non e' un numero valido e dentro il range.
def ask_value(var, prompt: str) -> float:
    lo, hi = variable_range(var)
    while True:
        raw = input(f"{prompt} [{lo:g}-{hi:g}]: ").strip().replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            print(f"  '{raw}' non e' un numero valido. Riprova.")
            continue
        if not (lo <= value <= hi):
            print(f"  Valore fuori range: deve essere tra {lo:g} e {hi:g}. Riprova.")
            continue
        return value

# Valida un valore gia' disponibile (modalita' da riga di comando). Esce con messaggio d'errore se fuori range.
def validate(var, value: float) -> float:
    lo, hi = variable_range(var)
    if not (lo <= value <= hi):
        name = var.name
        sys.exit(f"Errore: {name}={value:g} fuori range [{lo:g}, {hi:g}].")
    return value

# funzione per mettere nel sistema valori input e ottenere output
def run(temp: float, sun: float, fis) -> float:
    y = fis.infer({"temperature": temp, "sunshine": sun})           # fa inferenza dei valori in input e ricava output
    print(f"\nInput : temperatura = {temp:g} C,  sole = {sun:g} %")
    print(f"Output: turisti stimati = {y:.1f} %\n")
    return y


def main():
    fis = build_system()
    print_header(fis)

    args = sys.argv[1:]
    if len(args) == 2:
        # modalita' non interattiva: valori da riga di comando
        try:
            temp = float(args[0].replace(",", "."))
            sun = float(args[1].replace(",", "."))
        except ValueError:
            sys.exit("Errore: gli argomenti devono essere due numeri "
                     "(temperatura e sole).")
        temp = validate(fis.inputs["temperature"], temp)
        sun = validate(fis.inputs["sunshine"], sun)
        run(temp, sun, fis)
    elif len(args) == 0:
        # modalita' interattiva
        temp = ask_value(fis.inputs["temperature"], "Temperatura (gradi C)")
        sun = ask_value(fis.inputs["sunshine"], "Sole (%)")
        run(temp, sun, fis)
    else:
        sys.exit("Uso: python interactive.py [temperatura sole]")


if __name__ == "__main__":
    main()
