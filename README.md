# Fuzzy Logic Lab

Raccolta di progetti di studio sulla **logica fuzzy** e sui sistemi di *computational intelligence*, sviluppati per consolidare la teoria vista a lezione e per sperimentare implementazioni via via più avanzate.

Ogni sottoprogetto è **autonomo** (proprio README, proprio `requirements.txt`, propri test) e affianca al codice la teoria di riferimento, così che la repository serva sia da esercizio pratico sia da materiale consultabile.

## Filosofia

Due principi guidano il codice:

1. **Prima da zero, poi con le librerie.** Ogni concetto chiave è implementato una volta a mano in NumPy — per mostrare che la meccanica è compresa — e una volta con la libreria di riferimento (`scikit-fuzzy`), verificando che i risultati coincidano. L'implementazione manuale è didattica; quella con libreria è quella che si userebbe in produzione.
2. **Progressione visibile.** I progetti sono numerati in ordine di complessità crescente, dalla regola scritta a mano dall'esperto fino al sistema che apprende i propri parametri dai dati.

## Struttura

```
fuzzy-logic-lab/
├── 01_mamdani_fis/        # [BASE] Sistema di inferenza Mamdani
│   ├── src/               #   implementazione from-scratch + scikit-fuzzy
│   ├── notebooks/         #   notebook di esplorazione e visualizzazione
│   ├── tests/             #   test di coerenza (pytest)
│   └── figures/           #   grafici generati
├── docs/                  # approfondimenti teorici trasversali
├── LICENSE
└── README.md              # questo file
```

## Roadmap dei progetti

| # | Progetto | Livello | Concetti | Stato |
|---|----------|---------|----------|-------|
| 01 | **Mamdani FIS** | Base | Fuzzification, t-norm/t-conorm, inferenza Mamdani, defuzzification | ✅ Completo |
| 02 | **Fuzzy C-Means** | Intermedio | Clustering fuzzy, partizioni sfumate, generazione regole data-driven | 🔲 Da fare |
| 03 | **ANFIS** | Intermedio-avanzato | Rete adattiva, TSK, training ibrido (gradient + LSE) | 🔲 Da fare |
| 04 | **Genetic-Fuzzy System** | Avanzato | Ottimizzazione delle MF via algoritmi genetici | 🔲 Da fare |
| 05 | **Interval Type-2 FLS** | Avanzato | Incertezza sulla membership, type reduction | 🔲 Opzionale |

> Ogni progetto elenca nel proprio README le **possibili estensioni**, così che possa crescere nel tempo.

## Come iniziare

```bash
git clone <url-del-tuo-repo>
cd fuzzy-logic-lab/01_mamdani_fis
python -m venv .venv && source .venv/bin/activate   # opzionale ma consigliato
pip install -r requirements.txt
python src/example_tourists.py       # esegue l'esempio
pytest tests/                        # lancia i test
```

## Ecosistema Python usato

- **[scikit-fuzzy](https://scikit-fuzzy.github.io/scikit-fuzzy/)** — libreria di riferimento per la logica fuzzy nello stack SciPy; base dei progetti principali.
- **NumPy / SciPy / Matplotlib** — calcolo numerico e visualizzazione.
- Altre tecnologie provate nei singoli progetti come *mini-test* (es. `simpful` per il TSK, `pyit2fls` per il type-2, `DEAP` per gli algoritmi genetici) con le relative motivazioni d'uso documentate.

## Riferimenti teorici

I fondamenti seguono la letteratura standard: Zadeh (fuzzy sets), Mamdani (inferenza linguistica), Takagi–Sugeno–Kang (modelli TSK), Jang (ANFIS), Mendel (type-2). Il documento in `docs/` raccoglie approfondimenti trasversali.

## Licenza

MIT — vedi [LICENSE](LICENSE).
