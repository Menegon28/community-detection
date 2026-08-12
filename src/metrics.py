# Metriche di qualità (modularità, avg F1 e AMI vs ground truth) e misure di
# tempo, memoria di picco e volume di shuffle di Spark.

import time
import warnings
from collections import Counter, defaultdict

import networkx as nx
import pandas as pd
from memory_profiler import memory_usage
from sklearn.metrics import adjusted_mutual_info_score

# sopprime warning fastidioso
warnings.filterwarnings("ignore", message="The number of unique classes")


def partition_to_sets(partition):
    # converte {node: community_id} in una lista in cui ogni elemento
    # è un set contenente i nodi che appartengono ad una certa comunità
    groups = defaultdict(set)
    for node, cid in partition.items():
        groups[cid].add(node)
    return list(groups.values())


def prepare_ground_truth(gt_comms, graph_nodes):
    # prende in input la lista delle ground truth communities e il set dei nodi presenti
    # nel grafo (poiché potrebbe essere stato sottocampionato).

    # Restringe le community ai nodi presenti nel grafo
    restricted_comms = []
    for c in gt_comms:
        c = c & graph_nodes
        if c:
            restricted_comms.append(c)

    # Ricava le etichette nodo -> community (per l'AMIserve partizione non-overlapping),
    # ordina le comunità per grandezza, in modo che l'id finale per un nodo
    # sia quello della comunità più grande a cui appartiene
    gt_labels = {}
    for cid, comm in enumerate(sorted(restricted_comms, key=len)):
        for node in comm:
            gt_labels[node] = cid
    return restricted_comms, gt_labels


def best_match_f1(a_comms, b_comms):
    # media su a_comms del miglior F1 rispetto a una qualsiasi community di b;
    # l'indice sui node_to_b serve a considerare solo le coppie che si sovrappongono
    node_to_b = defaultdict(list)
    for j, b in enumerate(b_comms):
        for node in b:
            node_to_b[node].append(j)

    total = 0.0
    for a in a_comms:
        # overlaps[j] = |a intersecato b_comms[j]|
        overlaps = Counter(j for node in a for j in node_to_b.get(node, ()))
        best = 0.0
        for j, inter in overlaps.items():
            precision = inter / len(a)
            recall = inter / len(b_comms[j])
            best = max(best, 2 * precision * recall / (precision + recall))
        total += best
    return total / len(a_comms)


def avg_f1(pred_comms, gt_comms):
    # F1 medio di Yang-Leskovec, media del best-match F1 nelle due direzioni; le
    # previsioni sono ristrette ai nodi etichettati, perché la ground truth copre
    # solo una parte del grafo e il resto non deve contare come errore
    labeled = set()
    for c in gt_comms:
        labeled.update(c)

    restricted = []
    for c in pred_comms:
        c = c & labeled
        if c:
            restricted.append(c)
    return 0.5 * (best_match_f1(restricted, gt_comms) + best_match_f1(gt_comms, restricted))


def ami(partition, gt_labels):
    # calcolata sui nodi che hanno sia un'etichetta di ground truth sia una previsione
    true_labels = []
    pred_labels = []
    for node, cid in gt_labels.items():
        if node in partition:
            true_labels.append(cid)
            pred_labels.append(partition[node])
    return adjusted_mutual_info_score(true_labels, pred_labels)


def evaluate(G, partition, gt_comms, gt_labels):
    # crea le colonne di qualità delle community, per il .csv dei risultati
    pred_comms = partition_to_sets(partition)
    return {
        "n_communities": len(pred_comms),
        "modularity": round(nx.community.modularity(G, pred_comms), 4),
        "avg_f1": round(avg_f1(pred_comms, gt_comms), 4),
        "ami": round(ami(partition, gt_labels), 4),
    }


def timed(func):
    # esegue func() e ne restituisce il risultato con il tempo impiegato
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def measure(func, include_children=False):
    # esegue func(), restituisce risultato, tempo e memoria di picco aggiuntiva.
    # si sottrae una baseline perché memory_profiler misura la RAM occupata dal
    # processo in assoluto (include quello già in RAM); il tempo si
    # misura dentro il campionamento per escludere l'overhead di avvio di quest'ultimo.
    # include_children serve in Spark, dove la memoria vera sta nella JVM figlia.
    baseline_mb = memory_usage(max_usage=True, include_children=include_children)
    peak_mem_mb, (value, runtime) = memory_usage(lambda: timed(func), max_usage=True,
                                                 retval=True, include_children=include_children)
    return value, runtime, peak_mem_mb - baseline_mb


def shuffle_snapshot(spark):
    # MB di shuffle cumulati sugli stage già completati, letti dalla REST API della
    # Spark UI: la differenza prima/dopo dà il traffico di un algoritmo
    sc = spark.sparkContext
    url = f"{sc.uiWebUrl}/api/v1/applications/{sc.applicationId}/stages?status=complete"
    return pd.read_json(url)[["shuffleReadBytes", "shuffleWriteBytes"]].sum() / 1e6
