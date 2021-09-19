import numpy as np
import networkx as nx
from src.codebase.nodes import nodes_figure5
from matplotlib import pyplot as plt
import yaml
import os

# load project context
with open("project_context.yml", "r") as file:
    context = yaml.safe_load(file)
project_path = context["project_path"]


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


def getDigraphPathMetricsForHubs(Ax, inOutThresh=10):

    inDeg = np.sum(Ax, axis=1)
    outDeg = np.sum(Ax, axis=0)
    numHubs = len(np.where(inDeg >= inOutThresh)[0]) + len(
        np.where(outDeg >= inOutThresh)[0]
    )
    ttt = "num of hubs are " + str(numHubs)
    print(ttt)
    numHubs2 = len(
        np.union1d(
            np.where(inDeg >= inOutThresh)[0], np.where(outDeg >= inOutThresh)[0]
        )
    )
    ttt = "num of hubs2 are " + str(numHubs2)
    print(ttt)

    # bare minimum nodes should have greater than 1 in and out-degrees
    nonZeroIn = np.where(inDeg > 1)[0]
    nonZeroOut = np.where(outDeg > 1)[0]
    nonZeroInOut = np.intersect1d(nonZeroIn, nonZeroOut)

    inOutDeg = inDeg + outDeg
    indNodesTemp = np.where(inOutDeg >= inOutThresh)[0]
    indNodes2Use = np.intersect1d(nonZeroInOut, indNodesTemp)
    lenNodes2Use = len(indNodes2Use)

    G = convert_from_adj2networkX(Ax)

    paths = dict(nx.all_pairs_dijkstra_path(G))
    len_paths = dict(nx.all_pairs_dijkstra_path_length(G))

    # it is opposite from the adjacency matrix, i.e. pathsMatrix[i,j] is path length from i to j
    pathsMatrix = np.zeros((lenNodes2Use, lenNodes2Use))
    for indSource, source in enumerate(indNodes2Use):
        for target in len_paths[source].keys():
            if target in indNodes2Use:
                indTarget = np.where(indNodes2Use == target)[0][0]
                pathsMatrix[indSource, indTarget] = len_paths[source][target]
    numPaths = np.sum(pathsMatrix > 0)
    numNonPaths = lenNodes2Use * (lenNodes2Use - 1) - numPaths
    distPaths = pathsMatrix[np.where(pathsMatrix > 0)]

    return indNodes2Use, numPaths, numNonPaths, distPaths, pathsMatrix


def getDigraphPathMetricsInOutConvDivUnit(Ax, inOutThresh):

    inDeg = np.sum(Ax, axis=1)
    outDeg = np.sum(Ax, axis=0)

    nonZeroIn = np.where(inDeg > 1)[0]
    nonZeroOut = np.where(outDeg > 1)[0]
    nonZeroInOut = np.intersect1d(nonZeroIn, nonZeroOut)

    inOutDeg = inDeg + outDeg
    indNodesTemp = np.where(inOutDeg >= inOutThresh)[0]
    indNodesUnit = np.intersect1d(nonZeroInOut, indNodesTemp)
    lenNodesUnit = len(indNodesUnit)

    indAllNodes = np.arange(100)
    # the nodes outside the convergent-divergent unit
    indNodesOut = np.setdiff1d(indAllNodes, indNodesUnit, assume_unique=False)

    # measure path lengths
    G = convert_from_adj2networkX(Ax)
    paths = dict(nx.all_pairs_dijkstra_path(G))
    len_paths = dict(nx.all_pairs_dijkstra_path_length(G))

    # set up the matrix that will indicate the path from the out nodes to the unit and from the unit to the nodes
    nodes2unit = np.zeros((len(indNodesOut), len(indNodesUnit)))
    unit2nodes = np.zeros((len(indNodesUnit), len(indNodesOut)))

    # set up the matrix that will indicate the path from the out nodes to the unit
    # it is opposite from the adjacency matrix, i.e. nodes2unit[i,j] is path length from i to j
    nodes2unit = np.zeros((len(indNodesOut), len(indNodesUnit)))
    for indSource, source in enumerate(indNodesOut):
        for indTarget, target in enumerate(indNodesUnit):
            if target in len_paths[source].keys():
                tt = (
                    "indTarget is "
                    + str(indTarget)
                    + " and indSource is "
                    + str(indSource)
                )
                print(tt)
                nodes2unit[indSource, indTarget] = len_paths[source][target]

    # set up the matrix that will indicate the path from the unit to the out nodes
    unit2nodes = np.zeros((len(indNodesUnit), len(indNodesOut)))
    for indSource, source in enumerate(indNodesUnit):
        for indTarget, target in enumerate(indNodesOut):
            if target in len_paths[source].keys():
                tt = (
                    "indTarget is "
                    + str(indTarget)
                    + " and indSource is "
                    + str(indSource)
                )
                print(tt)
                unit2nodes[indSource, indTarget] = len_paths[source][target]

    numPathsIn = np.sum(nodes2unit > 0)
    numPathsOut = np.sum(unit2nodes > 0)
    numNonPathsIn = nodes2unit.shape[0] * nodes2unit.shape[1] - numPathsIn
    numNonPathsOut = unit2nodes.shape[0] * unit2nodes.shape[1] - numPathsOut
    distPathsIn = nodes2unit[np.where(nodes2unit > 0)]
    distPathsOut = unit2nodes[np.where(unit2nodes > 0)]

    # at least one of the nodes in the unit sending to node out
    Out = np.sum(unit2nodes, axis=0)

    # the node out sending to at least one of the nodes in the unit
    In = np.sum(nodes2unit, axis=1)

    return (
        indNodesUnit,
        indNodesOut,
        numPathsIn,
        numPathsOut,
        numNonPathsIn,
        numNonPathsOut,
        distPathsIn,
        distPathsOut,
        In,
        Out,
    )


def run_analysis(P_RAND, REP, IN_OUT_THRESH, P_R, FLAG, A):
    inOutDict = {}
    for rep in np.arange(REP):
        for indP, p in enumerate(P_RAND):
            Ax = A[rep + 1][p, 1][4000]

            # get directed graph path's metrics
            (
                indNodesUnit,
                indNodesOut,
                numPathsIn,
                numPathsOut,
                numNonPathsIn,
                numNonPathsOut,
                distPathsIn,
                distPathsOut,
                In,
                Out,
            ) = nodes_figure5.getDigraphPathMetricsInOutConvDivUnit(Ax, IN_OUT_THRESH)

            # store metrics data
            inOutDict[rep + 1, p, FLAG] = (
                indNodesUnit,
                indNodesOut,
                numPathsIn,
                numPathsOut,
                numNonPathsIn,
                numNonPathsOut,
                distPathsIn,
                distPathsOut,
                In,
                Out,
            )

    # initialize
    numConnectOut = {}
    numNoConnectOut = {}
    numConnectIn = {}
    numNoConnectIn = {}
    nodesAverage = {}

    for p in P_R:
        numConnectOut[p] = np.zeros((REP, 1))
        numNoConnectOut[p] = np.zeros((REP, 1))
        numConnectIn[p] = np.zeros((REP, 1))
        numNoConnectIn[p] = np.zeros((REP, 1))
        nodesAverage[p] = np.zeros((REP, 1))
        for rep in np.arange(REP):
            (
                indNodesUnit,
                indNodesOut,
                numPathsIn,
                numPathsOut,
                numNonPathsIn,
                numNonPathsOut,
                distPathsIn,
                distPathsOut,
                In,
                Out,
            ) = inOutDict[rep + 1, p, FLAG]

            numConnectOut[p][rep] = len(np.where(Out > 0)[0])
            numNoConnectOut[p][rep] = len(Out) - len(np.where(Out > 0)[0])
            numConnectIn[p][rep] = len(np.where(In > 0)[0])
            numNoConnectIn[p][rep] = len(In) - len(np.where(In > 0)[0])

            tts = (
                "For p = "
                + str(p)
                + " and iteration = "
                + str(rep)
                + " we have "
                + str(len(indNodesUnit))
                + " in the unit"
            )
            print(tts)
            nodesAverage[p][rep] = len(indNodesUnit)

        ttl = (
            "For p = "
            + str(p)
            + " the average number of nodes in the unit are "
            + str(np.mean(nodesAverage[p]))
        )
        print(ttl)
    return (
        inOutDict,
        numConnectOut,
        numNoConnectOut,
        numConnectIn,
        numNoConnectIn,
        nodesAverage,
    )


def convert_nodes_to_unit(REP, FLAG, inOutDict, p):

    # initialize
    allNumPathsIn = 0
    allNumNonPathsIn = 0
    allDistPathsIn = []
    allNumPathsOut = 0
    allNumNonPathsOut = 0
    allDistPathsOut = []
    numConnectOut = 0
    numNoConnectOut = 0
    numConnectIn = 0
    numNoConnectIn = 0
    nodesAverage = 0

    for rep in np.arange(REP):
        (
            indNodesUnit,
            indNodesOut,
            numPathsIn,
            numPathsOut,
            numNonPathsIn,
            numNonPathsOut,
            distPathsIn,
            distPathsOut,
            In,
            Out,
        ) = inOutDict[rep + 1, p, FLAG]

        # from nodes to unit
        allNumPathsIn += numPathsIn
        allNumNonPathsIn += numNonPathsIn
        for kk in np.arange(len(distPathsIn)):
            allDistPathsIn.append(distPathsIn[kk])

        # from unit to nodes
        allNumPathsOut += numPathsOut
        allNumNonPathsOut += numNonPathsOut
        for kk in np.arange(len(distPathsOut)):
            allDistPathsOut.append(distPathsOut[kk])
        numConnectOut += len(np.where(Out > 0)[0])
        numNoConnectOut += len(Out) - len(np.where(Out > 0)[0])
        numConnectIn += len(np.where(In > 0)[0])
        numNoConnectIn += len(In) - len(np.where(In > 0)[0])
        nodesAverage = nodesAverage + len(indNodesUnit)
    nodesAverage = nodesAverage / REP


def plot_panel_3(P_R, nodesAverage):
    nodesAverageAll = np.zeros((len(nodesAverage), 1))
    for ind, p in enumerate(P_R):
        nodesAverageAll[ind] = nodesAverage[p]

    # plot
    plt.plot(P_R, nodesAverageAll)
    plt.show()
    return p


def plot_panel_2(IN_OUT_THRESH, P_R, nodesAverage):
    nodesMean = np.zeros((len(nodesAverage), 1))
    nodesStd = np.zeros((len(nodesAverage), 1))
    markers = ["o", "v", ">", "<", "8", "s"]
    colors = ["blue", "orange", "green", "red"]
    for ind, p in enumerate(P_R):
        nodesMean[ind] = np.mean(nodesAverage[p])
        nodesStd[ind] = np.std(nodesAverage[p])
    plt.errorbar(P_R, nodesMean, color="grey", linewidth=2)
    for ind in np.arange(len(nodesMean)):
        plt.errorbar(
            P_R[ind],
            nodesMean[ind],
            nodesStd[ind],
            marker=markers[ind],
            color=colors[ind],
            linewidth=2,
        )
    plt.show()

    # save figure
    filePathPlot = os.path.join(
        project_path,
        "data/03_figures/convdiv/NumNodesInUnit" + str(IN_OUT_THRESH) + ".eps",
    )
    plt.savefig(filePathPlot, format="eps", dpi=1200)


def plot_panel_1(
    IN_OUT_THRESH, P_R, numConnectOut, numNoConnectOut, numConnectIn, numNoConnectIn
):
    percentIn = {}
    percentOut = {}
    for p in P_R:
        percentIn[p] = numConnectIn[p] / (numConnectIn[p] + numNoConnectIn[p])
        percentOut[p] = numConnectOut[p] / (numConnectOut[p] + numNoConnectOut[p])

    meanIn = {}
    stdIn = {}
    meanOut = {}
    stdOut = {}
    markers = ["o", "v", ">", "<", "8", "s"]
    for ind, p in enumerate(P_R):
        meanIn[p] = np.mean(percentIn[p])
        stdIn[p] = np.std(percentIn[p])
        meanOut[p] = np.mean(percentOut[p])
        stdOut[p] = np.std(percentOut[p])
        labelT = "p = " + str(p)
        plt.ylim([0.2, 1.1])
        plt.errorbar(
            [1, 2],
            [meanIn[p], meanOut[p]],
            [stdIn[p], stdOut[p]],
            marker=markers[ind],
            label=labelT,
            linewidth=2,
        )
        plt.xticks([1, 2])
        plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.show()

    # save figure
    filePathPlot = os.path.join(
        project_path,
        "data/03_figures/convdiv/percentInOutThresh" + str(IN_OUT_THRESH) + ".eps",
    )
    plt.savefig(filePathPlot, format="eps", dpi=1200)
