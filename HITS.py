import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from numpy import linalg as LA
import operator


def hits(graph, iter_count=500):
    nodes = graph.nodes()
    nodes_count = len(nodes)
    matrix = nx.to_numpy_matrix(graph, nodelist=nodes)

    hubs_score = np.ones(nodes_count)
    auth_score = np.ones(nodes_count)
    H = matrix * matrix.T
    A = matrix.T * matrix

    for i in range(iter_count):

        hubs_score = hubs_score * H
        auth_score = auth_score * A
        hubs_score = hubs_score / LA.norm(hubs_score)
        auth_score = auth_score / LA.norm(auth_score)
    hubs_score = np.array(hubs_score).reshape(-1,)
    auth_score = np.array(auth_score).reshape(-1,)
    hubs = dict(zip(nodes, hubs_score/2))
    authorities = dict(zip(nodes, auth_score/2))
    print("Authority:")
    Auth = []
    for i in authorities:
        Auth.append(authorities[i])
    print("[", end="")
    for j in range(0, len(Auth)-1):
        print(Auth[j], end=" ")
    print(Auth[-1], end="")
    print("]")

    Hubs = []
    print("Hub:")
    Hubs = []
    for i in hubs:
        Hubs.append(hubs[i])
    print("[", end="")
    for j in range(0, len(Hubs)-1):
        print(Hubs[j], end=" ")
    print(Hubs[-1], end="")
    print("]")

    return hubs, authorities


def build_graph(path):
    graph_data = pd.read_csv(path, header=None)
    graph_data = graph_data.values.tolist()
    G = nx.DiGraph()
    G.add_edges_from(graph_data)
    return G


def totuple(word):
    form = []
    for i in range(0, len(word)-1):
        form.append((int(word[i]), int(word[i+1])))
    return form


def graph_plot(G):
    plt.figure(figsize=(10, 10))
    nx.draw_networkx(G, with_labels=True)
    plt.show()


CWD = os.getcwd()
files = os.listdir(os.path.join(CWD, 'hw3dataset'))
files.pop(-1)  # Get rid of Readme.txt
# print(files)
i = 0  # for csv_filename
csv_filename = ['graph_1.csv', 'graph_2.csv', 'graph_3.csv',
                'graph_4.csv', 'graph_5.csv', 'graph_6.csv']
hub_str = 'hub_'
authorities_str = 'authorities_'
graph = build_graph(os.path.join(CWD, 'hw3dataset', 'graph_1.txt'))
hubs, authorities = hits(graph)


# np.savetxt('graph_1_HITS_hub.txt', Hubs, fmt="%.18f", delimiter=" ")
# -------------------IBM
'''simpDat = []
text = []
temp = []
f = open('data.ntrans_1.npats_10.nitems_1.tlen_5.txt')
for line in f:
    simpDat.append(line)
f.close()

for i in range(0, len(simpDat)):
    if(simpDat[i][5:10] == simpDat[i-1][5:10]):
        temp.extend([simpDat[i][-6:-1]])
    else:
        text.append(temp)
        temp = []
        temp.extend([simpDat[i][-6:-1]])
text = text[1:]
text.append(['  111', '  222', '  444', '  487', '  913'])

text_all = []
for i in range(0, len(text)):
    for j in range(0, len(text[i])):
        text_all.append(text[i][j])

array = totuple(text_all)
G = nx.DiGraph()
G.add_edges_from(array)
hubs, authorities = hits(G)
graph_plot(G)'''
