import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from time import *

# First step, create graph in order to perform hits and pagerank
def build_graph(path):
    graph_data = pd.read_csv(path,header=None)
    graph_data = graph_data.values.tolist()
    G = nx.DiGraph()
    G.add_edges_from(graph_data) 
    return G
    
def build_graph_with_list(list):
    G = nx.DiGraph()
    G.add_edges_from(list) 
    return G

# Perform hits algorithm and pagerank algorithm
def hitsandpr(G,damping_factors=0.15):
    startTime = time()
    hubs, authorities = nx.hits(G,normalized = True)
    print("HITS computation time ::" , time() - startTime)
    #print("Hub Scores: ", hubs) 
    #print("Authority Scores: ", authorities)
    startTime = time()
    pr=nx.pagerank(G,damping_factors,max_iter=500)
    print("PageRank computation time ::" , time() - startTime)
    #print("PageRank Values : ", pr)
    return hubs, authorities, pr

def graph_plot(G):
    plt.figure(figsize =(10, 10)) 
    nx.draw_networkx(G, with_labels = True)
    plt.show()
    
def main():
    CWD = os.getcwd()
    files = os.listdir(os.path.join(CWD,'hw3dataset'))
    files.pop(-1) # Get rid of Readme.txt
    files.pop(0)
    result_str = ['hub :: ','authorities :: ','pagerank values :: ']
    for i in range(3):
        files.pop(-1)
    print(files)
    for file in files:
        i = 0
        graph = build_graph(os.path.join(CWD,'hw3dataset',file))
        results = hitsandpr(graph)
        for result in results:
            print(result_str[i] + str(result))
            i+=1
    graph_new = pd.read_csv(os.path.join(CWD,'hw3dataset',files[0]),header=None)
    graph_list = graph_new.values.tolist()
    graph_list.append([2,1])
    print(graph_list)
    graph_new = build_graph_with_list(graph_list)
    i = 0
    results = hitsandpr(graph_new)
    for result in results:
        print(result_str[i] + str(result))
        i+=1
    #graph_plot(graph_new)
    '''
    print(graph_list)
    graph_list.append([1,3])
    print(graph_list)
    graph_new = build_graph_with_list(graph_list)
    i = 0
    results = hitsandpr(graph_new)
    for result in results:
        print(result_str[i] + str(result))
        i+=1
    graph_list.append([1,4])
    print(graph_list)
    graph_new = build_graph_with_list(graph_list)
    i = 0
    results = hitsandpr(graph_new)
    for result in results:
        print(result_str[i] + str(result))
        i+=1
    graph_list.append([1,5])
    print(graph_list)
    graph_new = build_graph_with_list(graph_list)
    i = 0
    results = hitsandpr(graph_new)
    for result in results:
        print(result_str[i] + str(result))
        i+=1
    graph_list.append([1,6])
    print(graph_list)
    graph_new = build_graph_with_list(graph_list)
    i = 0
    results = hitsandpr(graph_new)
    for result in results:
        print(result_str[i] + str(result))
        i+=1
    '''
    


if __name__ == "__main__":
    main()