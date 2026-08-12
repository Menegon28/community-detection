# Algoritmi di community detection: centralizzati (NetworkX) e distribuiti (GraphFrames)

import networkx as nx


def _labels_from_communities(communities):
    # NetworkX restituisce una lista di set di nodi: le convertiamo in {nodo: id_comunita}
    labels = {}
    for community_id, community in enumerate(communities):
        for node in community:
            labels[node] = community_id
    return labels


def local_louvain(G, seed=123):
    communities = nx.community.louvain_communities(G, seed=seed)
    return _labels_from_communities(communities)


def local_lpa(G, seed=123):
    communities = nx.community.asyn_lpa_communities(G, seed=seed)
    return _labels_from_communities(communities)


def spark_lpa(g, max_iter=5):
    # sincrono a iterazioni fisse, mentre local_lpa è asincrono fino a convergenza
    result = g.labelPropagation(maxIter=max_iter)

    # GraphFrames restituisce un DataFrame (id, label): lo portiamo
    # allo stesso formato {nodo: id_comunita} degli algoritmi locali
    labels = {}
    for row in result.select("id", "label").collect():
        labels[row["id"]] = row["label"]
    return labels
