# Genera le figure di analisi da results/raw/*.csv dentro results/figures/.
#
# uv run python -m src.benchmark.plots

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

FIGURES = Path("results/figures")
DATASETS = ["com-youtube", "com-lj"]
SPARK_LPA_ITER = 5  # --lpa-iter usato nelle run, non registrato nei CSV
SPARK_CORES = 4  # numero di core mostrato nei confronti con il locale

ALGO_COLOR = {"louvain": "tab:blue", "lpa": "tab:orange"}
DATASET_COLOR = {"com-youtube": "tab:blue", "com-lj": "tab:orange"}

plt.rcParams["figure.dpi"] = 150
plt.rcParams["axes.grid"] = True


def run_label(algorithm, backend):
    if backend == "spark":
        return f"{algorithm} (spark, {SPARK_LPA_ITER} iter, {SPARK_CORES} core)"
    return f"{algorithm} (locale)"


def load_runs():
    # tiene solo l'ultima run di ogni configurazione
    local = pd.read_csv("results/raw/local_runs.csv")
    local = local.drop_duplicates(["dataset", "fraction", "algorithm"], keep="last")
    local["backend"] = "local"
    spark = pd.read_csv("results/raw/spark_runs.csv")
    spark = spark.drop_duplicates(["dataset", "fraction", "cores", "algorithm"], keep="last")
    spark["backend"] = "spark"
    return local, spark


def plot_vs_size(local, spark, dataset, column, ylabel, filename):
    # una linea per algoritmo x backend, x = dimensione del grafo; Spark a SPARK_CORES
    fig, ax = plt.subplots(figsize=(7, 4.5))
    runs = {"local": local[local.dataset == dataset],
            "spark": spark[(spark.dataset == dataset) & (spark.cores == SPARK_CORES)]}
    for backend, df in runs.items():
        for algo, g in df.groupby("algorithm"):
            g = g.sort_values("n_edges")
            ax.plot(g.n_edges, g[column], marker="o", color=ALGO_COLOR[algo],
                    linestyle="--" if backend == "spark" else "-",
                    label=run_label(algo, backend))
    ax.set_xscale("log")
    ax.set_yscale("log")
    # aggiungo i tick 2/5 per avere più riferimenti
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("archi")
    ax.set_ylabel(ylabel)
    ax.set_title(dataset)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / filename)
    plt.close(fig)


def plot_speedup(local, spark):
    # runtime e speedup vs core (LPA Spark sui grafi completi); la linea orizzontale
    # è il runtime dell'LPA locale sullo stesso grafo, cioè il baseline da battere
    full = spark[spark.fraction == 1.0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    for dataset, g in full.groupby("dataset"):
        g = g.sort_values("cores")
        ax1.plot(g.cores, g.runtime_s, marker="o", color=DATASET_COLOR[dataset], label=dataset)
        ref = local[(local.dataset == dataset) & (local.fraction == 1.0)
                    & (local.algorithm == "lpa")].runtime_s.iloc[0]
        ax1.axhline(ref, color=DATASET_COLOR[dataset], linestyle=":",
                    label=f"{dataset} LPA locale")
        speedup = g.runtime_s.iloc[0] / g.runtime_s  # baseline = 1 core
        ax2.plot(g.cores, speedup, marker="o", color=DATASET_COLOR[dataset], label=dataset)
    cores = sorted(full.cores.unique())
    ax2.plot(cores, cores, color="gray", linestyle=":", label="ideale")
    ax1.set_yscale("log")
    ax1.set_xlabel("core")
    ax1.set_ylabel("runtime (s)")
    ax1.set_xticks(cores)  # solo i valori misurati, non 3 e 5
    ax1.legend()
    ax2.set_xlabel("core")
    ax2.set_ylabel("speedup vs 1 core")
    ax2.set_xticks(cores)
    ax2.legend()
    fig.suptitle("Scalabilità dell'LPA Spark (frazione 1.0)")
    fig.tight_layout()
    fig.savefig(FIGURES / "speedup_vs_cores.png")
    plt.close(fig)


def plot_shuffle(spark):
    # traffico di shuffle totale vs core, proxy dell'overhead di comunicazione
    full = spark[spark.fraction == 1.0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for dataset, g in full.groupby("dataset"):
        g = g.sort_values("cores")
        ax.plot(g.cores, g.shuffle_read_mb + g.shuffle_write_mb, marker="o",
                color=DATASET_COLOR[dataset], label=dataset)
    ax.set_xlabel("core")
    ax.set_xticks(sorted(full.cores.unique()))
    ax.set_ylabel("shuffle letto + scritto (MB)")
    ax.set_title("Traffico di shuffle dell'LPA Spark (frazione 1.0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "shuffle_vs_cores.png")
    plt.close(fig)


def plot_quality(local, spark, dataset):
    # barre per metrica, alla frazione più grande presente in entrambi i CSV
    local = local[local.dataset == dataset]
    spark = spark[spark.dataset == dataset]
    fraction = max(set(local.fraction) & set(spark.fraction))
    rows = pd.concat([
        local[local.fraction == fraction],
        spark[(spark.fraction == fraction) & (spark.cores == SPARK_CORES)],
    ]).set_index(["algorithm", "backend"])
    order = [("louvain", "local"), ("lpa", "local"), ("lpa", "spark")]

    metrics = [("modularity", "modularità"), ("ami", "AMI"), ("avg_f1", "avg F1")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    for ax, (col, title) in zip(axes, metrics):
        values = [rows.loc[key, col] for key in order]
        bars = ax.barh(range(len(order)), values, height=0.65,
                       color=[ALGO_COLOR[algo] for algo, _ in order])
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.15)
    # asse y condiviso, etichette e verso impostati una volta sola
    axes[0].set_yticks(range(len(order)), [run_label(algo, backend) for algo, backend in order])
    axes[0].invert_yaxis()
    fig.suptitle(f"{dataset}: qualità delle community (frazione {fraction})")
    fig.tight_layout()
    fig.savefig(FIGURES / f"{dataset}_quality.png")
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    local, spark = load_runs()
    for dataset in DATASETS:
        plot_vs_size(local, spark, dataset, "runtime_s", "runtime (s)",
                     f"{dataset}_runtime_vs_size.png")
        plot_vs_size(local, spark, dataset, "peak_mem_mb", "memoria di picco aggiuntiva (MB)",
                     f"{dataset}_memory_vs_size.png")
        plot_quality(local, spark, dataset)
    plot_speedup(local, spark)
    plot_shuffle(spark)
    print(f"figure scritte in {FIGURES}\\")


if __name__ == "__main__":
    main()
