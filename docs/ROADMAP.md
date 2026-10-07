# Roadmap dettagliata dei progetti

Questo documento descrive i progetti previsti, con obiettivi, concetti coperti e tecnologie suggerite. Serve da guida per lo sviluppo incrementale della repository.

---

## 01 — Mamdani FIS *(completo)*

**Obiettivo:** implementare un sistema di inferenza fuzzy Mamdani da zero e verificarlo contro scikit-fuzzy.

**Concetti:** fuzzification, t-norm/t-conorm, implicazione, aggregazione, defuzzification.

**Tecnologia:** NumPy + scikit-fuzzy. È la base stabile su cui poggiano gli altri progetti.

---

## 02 — Fuzzy C-Means

**Obiettivo:** clustering fuzzy su un dataset reale (es. Iris), con confronto rispetto a k-means.

**Concetti:** partizioni sfumate, fuzzifier `m`, minimizzazione alternata, generazione di regole data-driven dai centri dei cluster.

**Tecnologie:**
- Implementazione from-scratch in NumPy (didattica).
- `skfuzzy.cmeans` (versione di riferimento).
- `scikit-learn` per il k-means di confronto e le metriche.

**Estensioni:** usare i centri trovati per inizializzare le membership function di un FIS — ponte diretto verso ANFIS.

---

## 03 — ANFIS

**Obiettivo:** riprodurre l'approssimazione di una funzione non lineare (es. `sinc`) con una rete adattiva equivalente a un sistema TSK.

**Concetti:** architettura a 5 layer, training ibrido (least squares in forward + gradient in backward), premise e consequent parameters.

**Tecnologie (mini-test comparativo):**
- Una libreria ANFIS esistente per Python.
- Una versione in **PyTorch** con autograd, per mostrare l'equivalenza rete ↔ FIS e sfruttare la GPU. *Motivazione:* dataset grandi e integrazione in pipeline deep learning.

**Estensioni:** confrontare grid partitioning vs. clustering (dal progetto 02) per la generazione delle regole.

---

## 04 — Genetic-Fuzzy System

**Obiettivo:** ottimizzare i parametri delle membership function di un FIS (dal progetto 01) tramite un algoritmo genetico.

**Concetti:** codifica dei parametri MF nel cromosoma, fitness = −RMSE, selezione/crossover/mutazione, compromesso accuratezza–interpretabilità.

**Tecnologie:**
- GA implementato da zero (riusa la teoria sugli algoritmi genetici del corso).
- In alternativa **DEAP** per non riscrivere l'infrastruttura evolutiva. *Motivazione:* framework maturo per l'evolutionary computation.

**Estensioni:** ottimizzazione multi-obiettivo (accuratezza vs. numero di regole) con NSGA-II.

---

## 05 — Interval Type-2 FLS *(opzionale, avanzato)*

**Obiettivo:** costruire un controller di tipo 2 e confrontarlo con l'equivalente type-1 sotto rumore crescente.

**Concetti:** Footprint of Uncertainty, MF superiore/inferiore, type reduction (Karnik-Mendel).

**Tecnologia:** `pyit2fls`. *Motivazione:* unica libreria pratica per il type-2. È il progetto che più distingue il portfolio dal livello standard del corso.

---

## Principi trasversali

- **Ogni progetto** ha README con teoria, codice commentato, test e almeno una visualizzazione.
- **Non disperdersi:** meglio 3-4 progetti completi che molti abbozzati. I mini-test su librerie alternative restano sezioni brevi dentro i progetti principali.
- **Progressione leggibile:** la sequenza 01→04 racconta l'arco dalla regola scritta a mano al sistema che apprende dai dati.
