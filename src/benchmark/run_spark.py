# Esegue l'LPA distribuito su un dataset SNAP com-* e aggiunge i risultati a
# results/raw/spark_runs.csv.
#
# docker compose run --rm spark python -m src.benchmark.run_spark \
#     [--dataset com-youtube] [--fraction 1.0] [--seed 123] [--cores 2] [--lpa-iter 5]

import argparse
from pathlib import Path

import pandas as pd

from src.algorithms import spark_lpa
from src.ingestion import (
    get_spark,
    load_communities,
    load_graph,
    read_edges,
    to_graphframe,
)
from src.metrics import evaluate, measure, prepare_ground_truth, shuffle_snapshot

RESULTS_CSV = Path("results/raw/spark_runs.csv")


def main():
    parser = argparse.ArgumentParser(description="Esegue l'LPA distribuito su un dataset SNAP.")
    parser.add_argument("--dataset", default="com-youtube", help="prefisso del nome del dataset SNAP")
    parser.add_argument("--fraction", type=float, default=1.0, help="frazione di nodi da campionare")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cores", type=int, default=2, help="thread worker per local[N]")
    parser.add_argument("--lpa-iter", type=int, default=5)
    args = parser.parse_args()

    spark = get_spark(args.cores)
    edges_sdf, n_edges = read_edges(spark, args.dataset, args.fraction, args.seed, args.cores)
    print(f"grafo: {n_edges:,} archi "
          f"(dataset={args.dataset}, fraction={args.fraction}, cores={args.cores})")

    # costruisce il grafo fuori dal blocco misurato, così come load_graph() in run_local.py:
    # si misura solo il costo dell'algoritmo, non anche la lettura del parquet
    # e la costruzione del set di vertici
    g = to_graphframe(edges_sdf)
    g.vertices.cache()
    g.edges.cache()
    g.vertices.count()

    before = shuffle_snapshot(spark)
    partition, runtime, peak_mem = measure(
        lambda: spark_lpa(g, args.lpa_iter), include_children=True)
    shuffle_read, shuffle_write = (shuffle_snapshot(spark) - before).tolist()
    spark.stop()  # libera la JVM prima di costruire il grafo NetworkX qui sotto

    # stesso grafo in NetworkX per calcolare le metriche
    G = load_graph(args.dataset, args.fraction, args.seed)
    gt_comms, gt_labels = prepare_ground_truth(load_communities(args.dataset), set(G))

    quality = evaluate(G, partition, gt_comms, gt_labels)
    row = {
        "dataset": args.dataset,
        "fraction": args.fraction,
        "seed": args.seed,
        "cores": args.cores,
        "algorithm": "lpa",
        "n_nodes": len(partition),
        "n_edges": n_edges,
        "runtime_s": round(runtime, 2),
        "peak_mem_mb": round(peak_mem, 1),
        "shuffle_read_mb": round(shuffle_read, 1),
        "shuffle_write_mb": round(shuffle_write, 1),
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
