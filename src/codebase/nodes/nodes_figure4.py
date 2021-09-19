import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

def convert_from_adj2networkX(A, weight_d="binary"):

    edges_ind = np.where(A > 0)
    num_edges = len(edges_ind[0])

    G = nx.DiGraph()  # DiGraph
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

    paths = dict(nx.all_pairs_dijkstra_path(G))
    len_paths = dict(nx.all_pairs_dijkstra_path_length(G))

    total_inv_path_len = 0
    counter = 0
    for edge in len_paths.keys():
        for conn_edge in len_paths[edge].keys():

            counter += 1
            len_p = len_paths[edge][conn_edge]
            if len_p > 0:
                total_inv_path_len += 1 / len_p
            # else:
            #    print('edge is '+str(edge)+' and connected edge is '+str(conn_edge))

            # print('%d-%d: %f'%(edge,conn_edge,len_p))
            # print('counter is %d, length of the path is %f and total path length so far is %f '%(counter,len_p,total_path_len))
            # print('')

    average_inv_path_len = total_inv_path_len / total_connections
    num_of_possible_paths = counter
    # print('average path length is %f'%average_path_len)
    return average_inv_path_len, num_of_possible_paths
    # CHANGE SO THAT YOU INCLUDE THE DISCONNECTED NODES TOO. DO EFFICIENCY


def plot_fig4C(P_RAND, DEG_SEL, possible_pathsAll, connections, MS, LW, colorsPlot, shapePointNoLine, degAll):

    # calculate runtime parameters
    degSelInd = np.zeros(len(DEG_SEL))
    for ind, dS in enumerate(DEG_SEL):
        degSelInd[ind] = np.where(degAll == DEG_SEL[ind])[0][0]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for ind, dS in enumerate(DEG_SEL):
        ttl = str(dS)

        # get metric
        metric = connections[int(degSelInd[ind]), :, :] * possible_pathsAll

        # calculate summary statistics
        degStd = np.std(metric, axis=0)
        degMean = np.mean(metric, axis=0)
        ax.errorbar(
            P_RAND,
            degMean,
            degStd,
            mfc=colorsPlot[ind],
            mec=colorsPlot[ind],
            marker=shapePointNoLine[ind],
            markersize=MS,
            color=colorsPlot[ind],
            linewidth=LW,
            label=ttl,
        )
    plt.xlabel("Prandom")
    plt.ylabel("Structure efficiency (available paths)")
    plt.show()

def plot_fig4B(P_RAND, DEG_SEL, path_lens, connections, MS, LW, colorsPlot, shapePointNoLine, degAll):
    
    # plot Fig 4B =================
    # calculate runtime parameters
    degSelInd = np.zeros(len(DEG_SEL))
    for ind, dS in enumerate(DEG_SEL):
        degSelInd[ind] = np.where(degAll == DEG_SEL[ind])[0][0]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for ind, dS in enumerate(DEG_SEL):
        ttl = str(dS)

        # get metric
        metric = connections[int(degSelInd[ind]), :, :] / path_lens

        # calculate summary statistics
        degStd = np.std(metric, axis=0)
        degMean = np.mean(metric, axis=0)

        # plot
        ax.errorbar(
            P_RAND,
            degMean,
            degStd,
            mfc=colorsPlot[ind],
            mec=colorsPlot[ind],
            marker=shapePointNoLine[ind],
            markersize=MS,
            color=colorsPlot[ind],
            linewidth=LW,
            label=ttl,
        )
    plt.xlabel("Prandom")
    plt.ylabel("Structure efficiency (path length)")
    plt.show()


def plot_fig4A(
    MIN_CON, MAX_CON, P_RAND, DEG_SEL, connections, MS, LW, colorsPlot, shapePointNoLine
):

    # calculate runtime parameters
    len_min_max = MAX_CON - MIN_CON + 1
    degAll = np.arange(MIN_CON, MAX_CON + 1)
    degSelInd = np.zeros(len(DEG_SEL))
    for ind, dS in enumerate(DEG_SEL):
        degSelInd[ind] = np.where(degAll == DEG_SEL[ind])[0][0]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for ind, dS in enumerate(DEG_SEL):
        ttl = str(dS)

        # calculate summary stats
        degStd = np.std(connections[int(degSelInd[ind]), :, :], axis=0)
        degMean = np.mean(connections[int(degSelInd[ind]), :, :], axis=0)

        # plot
        ax.errorbar(
            P_RAND,
            degMean,
            degStd,
            mfc=colorsPlot[ind],
            mec=colorsPlot[ind],
            marker=shapePointNoLine[ind],
            markersize=MS,
            color=colorsPlot[ind],
            linewidth=LW,
            label=ttl,
        )
        plt.legend()
    plt.xlabel("Prandom")
    plt.ylabel("Number of nodes above degree threshold")
    plt.show()
    return degAll


def run_analysis(MIN_CON, MAX_CON, P_RAND, REP, A):

    # run analysis
    path_lens = np.zeros((REP, len(P_RAND)))
    possible_pathsAll = np.zeros((REP, len(P_RAND)))
    len_min_max = MAX_CON - MIN_CON + 1
    connections = np.zeros((len_min_max, REP, len(P_RAND)))

    for rep in np.arange(REP):
        for indP, p in enumerate(P_RAND):
            Ax = A[rep + 1][p, 1][4000]
            G = convert_from_adj2networkX(Ax)
            (
                average_inv_path_len,
                num_of_possible_paths,
            ) = get_inv_path_length(G)

            # get path lengths
            path_lens[rep, indP] = 1 / average_inv_path_len

            # get number of possible paths
            possible_pathsAll[rep, indP] = num_of_possible_paths

            # get in and out degrees
            degOut = np.sum(Ax, axis=0)
            degIn = np.sum(Ax, axis=1)

            for indT, thresh in enumerate(np.arange(MIN_CON, MAX_CON + 1)):
                connections[indT, rep, indP] = len(
                    np.union1d(
                        np.where(degIn >= thresh)[0], np.where(degOut >= thresh)[0]
                    )
                )

    return path_lens, possible_pathsAll, connections

