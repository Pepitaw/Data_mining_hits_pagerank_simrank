import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from time import *

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

# Used to plot graph


def graph_plot(G):
    plt.figure(figsize=(10, 10))
    nx.draw_networkx(G, with_labels=True)
    plt.show()


def main():
    CWD = os.getcwd()
    files = os.listdir(os.path.join(CWD, 'hw3dataset'))
    files.pop(-1)  # Get rid of Readme.txt
    print(files)
    i = 0  # for csv_filename
    csv_filename = ['graph_1.csv', 'graph_2.csv', 'graph_3.csv',
                    'graph_4.csv', 'graph_5.csv', 'graph_6.csv']
    hub_str = 'hub_'
    authorities_str = 'authorities_'
    pr_str = 'pr_'
    damping_list = np.arange(0.15, 0.95, 0.1, dtype=np.float32)

    graph = build_graph(os.path.join(CWD, 'hw3dataset', 'graph_1.txt'))
    results = hitsandpr(graph, 0.15)
    print(results)
    '''
    # This part is for the graph 6
    graph = build_graph(os.path.join(CWD,'hw3dataset',files[-1]))
    startTime = time()
    hubs, authorities = nx.hits(graph, max_iter=120, normalized=True)
    print("HITS computation time ::" , time() - startTime)
    #print(hubs)
    #print(authorities)
    a = pd.DataFrame.from_dict(hubs, orient='index')
    a.to_csv(hub_str + csv_filename[-1])
    a = pd.DataFrame.from_dict(authorities, orient='index')
    a.to_csv(authorities_str + csv_filename[-1])
    startTime = time()
    pr=nx.pagerank(graph,0.15)
    print("PageRank computation time ::" , time() - startTime)
    #print("PageRank Values : ", pr)
    a = pd.DataFrame.from_dict(pr, orient='index')
    a.to_csv(pr_str + csv_filename[-1])
    '''


if __name__ == "__main__":
    main()
