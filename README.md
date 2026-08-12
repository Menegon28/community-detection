# Benchmarking di Community Detection (progetto MIDST)

Confronta community detection **centralizzata** (NetworkX) e **distribuita**
(PySpark/GraphFrames) su grafi SNAP:

- **Label Propagation**: locale (NetworkX) vs distribuito (GraphFrames)
- **Louvain**: solo locale, come benchmark di qualità

Ogni run carica il grafo (o una frazione campionata dei nodi), esegue
l'algoritmo misurando tempo e memoria di picco, ne valuta la qualità
(modularità + avg_f1/AMI rispetto alla ground truth) e aggiunge una riga a
`results/raw/`. Da quei CSV `plots.py` genera le figure di confronto: runtime e
memoria al crescere della dimensione, qualità, speedup e shuffle al variare dei
core Spark.

Dataset usati: `com-youtube` (predefinito) e `com-lj` (~12 volte gli archi),
per vedere dove l'esecuzione centralizzata smette di scalare.

## Dati

Scaricare i file della sezione **Files** da:

- https://snap.stanford.edu/data/com-Youtube.html
- https://snap.stanford.edu/data/com-LiveJournal.html

estrarli e posizionarli in `data/raw/` mantenendo i nomi originali
(`com-youtube.ungraph.txt`, `com-youtube.top5000.cmty.txt`, ecc.).

## Setup

Prerequisiti: [uv](https://docs.astral.sh/uv/) e
[Docker](https://docs.docker.com/get-docker/).

```powershell
uv sync               # per l'esecuzione locale (NetworkX)
docker compose build  # per l'esecuzione distribuita (PySpark/GraphFrames)
```

## Esecuzione

**Locale**:

```powershell
uv run python -m src.benchmark.run_local --algorithm louvain|lpa `
    [--dataset com-youtube|com-lj] [--fraction F] [--seed S]
```

**Distribuito**:

```powershell
docker compose run --rm spark python -m src.benchmark.run_spark `
    [--dataset com-youtube|com-lj] [--fraction F] [--seed S] [--cores N] [--lpa-iter I]
```

**Figure** (legge tutti i CSV e scrive in `results/figures/`):

```powershell
uv run python -m src.benchmark.plots
```
