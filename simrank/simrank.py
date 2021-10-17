import networkx as nx
import os
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt


def simrank(G, r, max_iter=100, eps=1e-4):
    nodes = list(G.nodes())
    nodes_i = {k: v for(k, v) in [(nodes[i], i) for i in range(0, len(nodes))]}

    sim_prev = np.zeros(len(nodes))
    sim = np.identity(len(nodes))

    for i in range(max_iter):
        if np.allclose(sim, sim_prev, atol=eps):
            break
        sim_prev = np.copy(sim)
        for u, v in itertools.product(nodes, nodes):
            if u is v:
                continue
            u_ns, v_ns = G.predecessors(u), G.predecessors(v)

            if (len(list(u_ns)) == 0 or len(list(v_ns)) == 0):
                sim[nodes_i[u]][nodes_i[v]] = 0
            elif(len(list(u_ns)) == 0 and len(list(v_ns)) == 0):
                sim[nodes_i[u]][nodes_i[v]] = 0
            else:
                s_uv = sum([sim_prev[nodes_i[u_n]][nodes_i[v_n]]
                            for u_n, v_n in itertools.product(u_ns, v_ns)])
                sim[nodes_i[u]][nodes_i[v]] = (
                    r * s_uv) / (len(list(u_ns)) * len(list(v_ns)))

    return sim

# First step, create graph in order to perform hits and pagerank


def build_graph(path):
    graph_data = pd.read_csv(path, header=None)
    graph_data = graph_data.values.tolist()
    G = nx.DiGraph()
    G.add_edges_from(graph_data)
    return G

# Perform hits algorithm and pagerank algorithm


def hitsandpr(G, damping_factors=0.15):
    startTime = time()
    hubs, authorities = nx.hits(G, normalized=True)
    print("HITS computation time ::", time() - startTime)
    #print("Hub Scores: ", hubs)
    #print("Authority Scores: ", authorities)
    startTime = time()
    pr = nx.pagerank(G, damping_factors, max_iter=500)
    print("PageRank computation time ::", time() - startTime)
    #print("PageRank Values : ", pr)
    return hubs, authorities, pr


def graph_plot(G):
    plt.figure(figsize=(10, 10))
    nx.draw_networkx(G, with_labels=True)
    plt.show()


def main():

    Decay_Factor = 0.8

    csv_filename = ['graph_1.csv', 'graph_2.csv',
                    'graph_3.csv', 'graph_4.csv', 'graph_5.csv']
    CWD = os.getcwd()
    files = os.listdir(os.path.join(CWD, '..\\hw3dataset'))
    files.pop(-1)  # Get rid of Readme.txt
    files.pop(0)
    files.pop(-1)
    # print(files)

    graph = build_graph(os.path.join(CWD, '..\hw3dataset', 'graph_1.txt'))
    # graph_plot(graph)
    s = simrank(graph, Decay_Factor)
    print(s)


if __name__ == "__main__":
    main()
