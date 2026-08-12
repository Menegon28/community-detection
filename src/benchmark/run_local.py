# Esegue un algoritmo di community detection centralizzato su un dataset SNAP com-* e
# aggiunge il risultato a results/raw/local_runs.csv.
#
# uv run python -m src.benchmark.run_local --algorithm louvain [--dataset com-youtube] [--fraction 1.0] [--seed 123]

import argparse
from pathlib import Path

import pandas as pd

from src.algorithms import local_louvain, local_lpa
from src.ingestion import load_communities, load_graph
from src.metrics import evaluate, measure, prepare_ground_truth

RESULTS_CSV = Path("results/raw/local_runs.csv")
ALGORITHMS = {"louvain": local_louvain, "lpa": local_lpa}


def main():
    parser = argparse.ArgumentParser(description="Esegue un algoritmo centralizzato su un dataset SNAP.")
    parser.add_argument("--algorithm", required=True, choices=list(ALGORITHMS))
    parser.add_argument("--dataset", default="com-youtube", help="prefisso del nome del dataset SNAP")
    parser.add_argument("--fraction", type=float, default=1.0, help="frazione di nodi da campionare")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    G = load_graph(args.dataset, args.fraction, args.seed)
    print(f"grafo: {G.number_of_nodes():,} nodi, {G.number_of_edges():,} archi "
          f"(dataset={args.dataset}, fraction={args.fraction})")

    gt_comms, gt_labels = prepare_ground_truth(load_communities(args.dataset), set(G))
    print(f"ground truth: {len(gt_comms):,} community, {len(gt_labels):,} nodi etichettati")

    algorithm = ALGORITHMS[args.algorithm]
    partition, runtime, peak_mem = measure(lambda: algorithm(G, seed=args.seed))
    quality = evaluate(G, partition, gt_comms, gt_labels)
    row = {
        "dataset": args.dataset,
        "fraction": args.fraction,
        "seed": args.seed,
        "algorithm": args.algorithm,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "runtime_s": round(runtime, 2),
        "peak_mem_mb": round(peak_mem, 1),
        "n_communities": quality["n_communities"],
        "modularity": quality["modularity"],
        "avg_f1": quality["avg_f1"],
        "ami": quality["ami"],
    }
    print(row)

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(RESULTS_CSV, mode="a", header=not RESULTS_CSV.exists(), index=False)
    print(f"aggiunta 1 riga a {RESULTS_CSV}")


if __name__ == "__main__":
    main()
