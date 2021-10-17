import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


def pagerank(graph, d, max_iter=500):
    W = nx.stochastic_graph(graph, weight='weight')
    N = W.number_of_nodes()
    x = dict.fromkeys(W, 1.0 / N)
    p = dict.fromkeys(W, 1.0 / N)

    for _ in range(max_iter):
        xlast = x
        # {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        x = dict.fromkeys(xlast.keys(), 0)
        for n in x:
            for nbr in W[n]:
                x[nbr] += (1-d) * xlast[n] * W[n][nbr]['weight']
            x[n] += d * p[n]

    pr_sorted = sorted(
        x.items(), key=lambda v: (v[1], v[0]), reverse=True)
    print("PageRank:")
    PR = []
    for i in x:
        PR.append(x[i])
    print("[", end="")
    for j in range(0, len(PR)-1):
        print(PR[j], end=" ")
    print(PR[-1], end="")
    print("]")
    return x


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


Damping_Factor = 0.15

CWD = os.getcwd()
files = os.listdir(os.path.join(CWD, 'hw3dataset'))
files.pop(-1)  # Get rid of Readme.txt
# print(files)
i = 0  # for csv_filename
csv_filename = ['graph_1.csv', 'graph_2.csv', 'graph_3.csv',
                'graph_4.csv', 'graph_5.csv', 'graph_6.csv']
pr_str = 'pr_'
damping_list = np.arange(0.15, 0.95, 0.1, dtype=np.float32)
graph = build_graph(os.path.join(CWD, 'hw3dataset', 'graph_1.txt'))
pr = pagerank(graph, Damping_Factor, max_iter=100)

# np.savetxt('graph_1_PageRank.txt', Hubs, fmt="%.18f", delimiter=" ")

# ----------------IBM_data
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
graph_plot(G)
pr = pagerank(G, 0.15, max_iter=100)'''

# --------------------increase---------------
'''graph_data = pd.read_csv('hw3dataset\graph_3.txt', header=None)
graph_data = graph_data.values.tolist()
graph_data.append([3, 1])
graph_data.append([6, 1])

G = nx.DiGraph()
G.add_edges_from(graph_data)
pr = pagerank(G, Damping_Factor, max_iter=100)'''
