# Lettura dei file SNAP, campionamento e conversione verso NetworkX e Spark/GraphFrames.
# Gli import di pyspark/graphframes sono all'interno delle funzioni,
# così da venir caricati solo per le esecuzioni in Spark, non per quelle in locale.

from pathlib import Path

import networkx as nx
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def read_snap(dataset):
    # archi normalizzati, senza self-loop né duplicati (imponiamo source < destination)
    df = pd.read_csv(RAW_DIR / f"{dataset}.ungraph.txt",
                     sep="\t", comment="#", names=["src", "dst"], dtype="int64")
    df = pd.DataFrame({"src": df.min(axis=1), "dst": df.max(axis=1)})
    return df[df.src != df.dst].drop_duplicates(ignore_index=True)


def sample_subgraph(edges, fraction, seed):
    # sottografo indotto da una frazione campionata casualmente di nodi,
    # di cui manteniamo solo la componente connessa più grande (cosiddetta "giant component")
    # in modo che i tempi di calcolo non dipendano da frammenti minuscoli
    nodes = pd.Series(pd.unique(edges[["src", "dst"]].values.ravel()))
    keep = nodes.sample(frac=fraction, random_state=seed)
    sub = edges[edges.src.isin(keep) & edges.dst.isin(keep)]
    giant = max(nx.connected_components(nx.from_pandas_edgelist(sub, "src", "dst")), key=len)
    return sub[sub.src.isin(giant) & sub.dst.isin(giant)].reset_index(drop=True)


def edges_parquet(dataset, fraction=1.0, seed=123):
    # cache parquet degli archi (grafo intero o campione di nodi),
    # costruita al primo accesso e condivisa dalle due pipeline
    if fraction == 1.0:
        path = PROCESSED_DIR / f"{dataset}.edges.parquet"
    else:
        path = PROCESSED_DIR / f"{dataset}.edges_f{fraction}_s{seed}.parquet"

    if not path.exists():
        if fraction == 1.0:
            edges = read_snap(dataset)
        else:
            # crea il parquet per fraction==1.0 se non esiste già, lo legge ed effettua il sottocampionamento
            edges = sample_subgraph(pd.read_parquet(edges_parquet(dataset)), fraction, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        edges.to_parquet(path, index=False)
    # ritorniamo la path, non i dati, così potrà essere letta o da pandas o da spark
    return path


def load_communities(dataset):
    # community di ground truth: ritorna una lista in cui ogni elemento è un set
    # contenente gli id dei nodi che appartengono a quella comunità.
    # si noti che le comunità possono sovrapporsi parzialmente
    communities = []
    with open(RAW_DIR / f"{dataset}.top5000.cmty.txt") as f:
        for line in f:
            # se line.strip() è una stringa vuota, continua
            if not line.strip():
                continue
            node_ids = {int(node_id) for node_id in line.split()}
            communities.append(node_ids)
    return communities


def load_graph(dataset, fraction, seed):
    # costruisce grafo NetworkX dal dataset (o da un suo campione)
    edges = pd.read_parquet(edges_parquet(dataset, fraction, seed))
    return nx.from_pandas_edgelist(edges, "src", "dst")


def get_spark(cores):
    # pseudo-cluster locale con `cores` workers (tutti dentro una sola JVM)
    from pyspark.sql import SparkSession

    # crea sessione spark, definisce numero di cores
    return (SparkSession.builder.appName("midst").master(f"local[{cores}]")
            # scarica il jar Scala di GraphFrames (l'algoritmo vero e proprio gira sulla JVM,
            # graphframes-py è solo il wrapper Python); fissa Spark3/Scala 2.12
            .config("spark.jars.packages", "io.graphframes:graphframes-spark3_2.12:0.12.1")
            .config("spark.driver.memory", "8g")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


def read_edges(spark, dataset, fraction, seed, cores):
    # legge gli archi dalla stessa cache parquet usata dagli algoritmi locali
    edges = spark.read.parquet(str(edges_parquet(dataset, fraction, seed)))
    n_edges = edges.count()

    # definisce il numero di partizioni che vanno poi distribuite tra i cores.
    # Inizialmente seguiva 4 * n. di cores, ma su com-lj ciò crea partizioni troppo grandi,
    # per cui è stato imposto un limite di 500k archi per partizione.
    # Le partizioni saranno assegnate al primo worker libero, una dopo l'altra
    spark.conf.set("spark.sql.shuffle.partitions", max(cores * 4, n_edges // 500_000))
    return edges, n_edges


def to_graphframe(edges_df):
    # converte uno Spark DF in un GraphFrame. Prende in input un DF di archi (generato da read_edges()) e
    # crea il corrispondente DF di nodi per passarli a GraphFrame().
    from graphframes import GraphFrame

    v = edges_df.select("src").union(edges_df.select("dst")).distinct().toDF("id")
    return GraphFrame(v, edges_df)
