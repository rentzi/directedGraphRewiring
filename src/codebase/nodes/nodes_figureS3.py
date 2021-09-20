import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

import logging

# configure logs
logging.basicConfig(
    filename="./logs/log.txt", format="%(asctime)s - %(message)s", level=logging.INFO
)


def convert_from_adj2networkX(A, weight_d="binary"):

    edges_ind = np.where(A > 0)
    num_edges = len(edges_ind[0])
    G = nx.DiGraph()
    G.add_nodes_from(np.arange(A.shape[0]))

    edges_list = list()
    if weight_d == "binary":
        for ind in np.arange(num_edges):
            edge_pair = (edges_ind[1][ind], edges_ind[0][ind])
            edges_list.append(edge_pair)

        G.add_edges_from(edges_list)
    else:
        for ind in np.arange(num_edges):
            edge_pair_w = (
                edges_ind[1][ind],
                edges_ind[0][ind],
                A[edges_ind[0][ind], edges_ind[1][ind]],
            )
            edges_list.append(edge_pair_w)
        G.add_weighted_edges_from(edges_list)
    return G


def get_inv_path_length(G):
    nodes = len(G.nodes)
    total_connections = nodes * (nodes - 1)
    len_paths = dict(nx.all_pairs_dijkstra_path_length(G))
    total_inv_path_len = 0
    counter = 0

    for edge in len_paths.keys():
        for conn_edge in len_paths[edge].keys():
            counter += 1
            len_p = len_paths[edge][conn_edge]
            if len_p > 0:
                total_inv_path_len += 1 / len_p
    average_inv_path_len = total_inv_path_len / total_connections
    num_of_possible_paths = counter
    return average_inv_path_len, num_of_possible_paths


def run_analysis(A, REP, P_ADV):
    path_lens = np.zeros((REP, len(P_ADV)))
    possible_pathsAll = np.zeros((REP, len(P_ADV)))

    for rep in np.arange(REP):
        for indP, p in enumerate(P_ADV):
            Ax = A[rep + 1][p, 1][4000]
            G = convert_from_adj2networkX(Ax)
            (average_inv_path_len, num_of_possible_paths) = get_inv_path_length(G)
            path_lens[rep, indP] = 1 / average_inv_path_len
            possible_pathsAll[rep, indP] = num_of_possible_paths
    return path_lens, possible_pathsAll


def plot_panel_2(P_ADV, possible_pathsAll):
    path_std = np.std(possible_pathsAll, axis=0)
    path_mean = np.mean(possible_pathsAll, axis=0)
    plt.errorbar(P_ADV, path_mean, path_std, marker="o", color="g", linewidth=2)
    plt.xlabel("Padvection")
    plt.ylabel("Number of pairs with a path")
    plt.show()


def plot_panel_1(P_ADV, path_lens):
    path_lens_std = np.std(path_lens, axis=0)
    path_lens_mean = np.mean(path_lens, axis=0)
    plt.errorbar(
        P_ADV, path_lens_mean, path_lens_std, marker="d", color="r", linewidth=2
    )
    plt.xlabel("Padvection")
    plt.ylabel("Path length")
    plt.show()
