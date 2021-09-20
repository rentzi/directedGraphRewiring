# Import dependencies
import numpy as np
import networkx as nx


def convert_from_adj2networkX(A, weight_d=str):
    """Convert from adjacency matrix to network graph

    Args:
        A ([type]): [description]
        weight_d (str, optional): e.g., "binary".

    Returns:
        [type]: [description]
    """
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


def getDigraphPathMetrics(Ax):
    """Get directed graph's path metrics

    Args:
        Ax ([type]): [description]

    Returns:
        [type]: [description]
    """
    G = convert_from_adj2networkX(Ax)
    nodes = len(G.nodes)
    len_paths = dict(nx.all_pairs_dijkstra_path_length(G))
    # it is opposite from the adjacency matrix, i.e.
    # pathsMatrix[i,j] is path length from i to j
    pathsMatrix = np.zeros((nodes, nodes))

    for source in np.arange(nodes):
        for target in len_paths[source].keys():
            pathsMatrix[source, target] = len_paths[source][target]
    numPaths = np.sum(pathsMatrix > 0)
    numNonPaths = nodes * (nodes - 1) - (numPaths + nodes)
    distPaths = pathsMatrix[np.where(pathsMatrix > 0)]
    invDistPaths = 1.0 / distPaths
    avInvPathAll = np.sum(invDistPaths) / (nodes * (nodes - 1))
    avPathAll = 1.0 / avInvPathAll
    avInvPathOnlyPaths = np.sum(invDistPaths) / numPaths
    avPathOnlyPaths = 1.0 / avInvPathOnlyPaths
    return numPaths, numNonPaths, distPaths, pathsMatrix, avPathAll, avPathOnlyPaths


def getDigraphCycleMetrics(pathsMatrix):
    """Get directed graph's cycle metrics

    Args:
        pathsMatrix ([type]): [description]

    Returns:
        [type]: [description]
    """
    nodes = pathsMatrix.shape[0]
    loopsMatrix = np.zeros((nodes, nodes))
    for a in np.arange(nodes):
        for b in np.arange(a + 1, nodes):
            cycleLen = 0
            if pathsMatrix[a, b] > 0:
                cycleLen += pathsMatrix[a, b]
                if pathsMatrix[b, a] > 0:
                    cycleLen += pathsMatrix[b, a]
                    loopsMatrix[a, b] = cycleLen
    numCycles = np.sum(loopsMatrix > 0)
    numNonCycles = nodes * (nodes - 1) / 2 - (numCycles)
    distCycles = loopsMatrix[np.where(loopsMatrix > 0)]
    return numCycles, numNonCycles, distCycles, loopsMatrix


def calculate_graph_metrics(pR, repetitions, FLAG, A):
    """Calculate directed graph's metrics

    Args:
        pR ([type]): [description]
        repetitions ([type]): [description]
        FLAG ([type]): [description]
        A ([type]): [description]

    Returns:
        [type]: [description]
    """
    pathDict = {}
    cycleDict = {}

    for rep in np.arange(repetitions):
        for indP, p in enumerate(pR):
            # calculate path metrics
            Ax = A[rep + 1][p, 1][4000]
            (
                numPaths,
                numNonPaths,
                distPaths,
                pathsMatrix,
                avPathAll,
                avPathOnlyPaths,
            ) = getDigraphPathMetrics(Ax)

            # calculate cycle metrics
            pathDict[rep + 1, p, FLAG] = (
                numPaths,
                numNonPaths,
                distPaths,
                pathsMatrix,
                avPathAll,
                avPathOnlyPaths,
            )
            (
                numCycles,
                numNonCycles,
                distCycles,
                loopsMatrix,
            ) = getDigraphCycleMetrics(pathsMatrix)
            cycleDict[rep + 1, p, FLAG] = (
                numCycles,
                numNonCycles,
                distCycles,
                loopsMatrix,
            )
    return pathDict, cycleDict


def calculate_aggregate_metrics(pR, repetitions, FLAG, pathDict):
    """Calculate aggregate metrics

    Args:
        pR ([type]): [description]
        repetitions ([type]): [description]
        FLAG ([type]): [description]
        pathDict ([type]): [description]

    Returns:
        [type]: [description]
    """
    # initialize
    pathOnlyPaths = np.zeros((len(pR), repetitions))
    pathAll = np.zeros((len(pR), repetitions))
    percNoPath = np.zeros((len(pR), repetitions))

    # run analysis over repetitions and pr
    for indP, p in enumerate(pR):
        for rep in np.arange(repetitions):
            (
                numPaths,
                numNonPaths,
                distPaths,
                pathsMatrix,
                avPathAll,
                avPathOnlyPaths,
            ) = pathDict[rep + 1, p, FLAG]

            pathAll[indP, rep] = avPathAll
            pathOnlyPaths[indP, rep] = avPathOnlyPaths
            percNoPath[indP, rep] = numNonPaths / (numNonPaths + numPaths)

    pathAllMean = np.zeros((len(pR), 1))
    pathAllStd = np.zeros((len(pR), 1))
    pathOnlyPathsMean = np.zeros((len(pR), 1))
    pathOnlyPathsStd = np.zeros((len(pR), 1))
    percNoPathMean = np.zeros((len(pR), 1))
    percNoPathStd = np.zeros((len(pR), 1))

    for ind, p in enumerate(pR):
        # path all
        pathAllMean[ind] = np.mean(pathAll[ind, :])
        pathAllStd[ind] = np.std(pathAll[ind, :])

        # path only
        pathOnlyPathsMean[ind] = np.mean(pathOnlyPaths[ind, :])
        pathOnlyPathsStd[ind] = np.std(pathOnlyPaths[ind, :])

        # no path
        percNoPathMean[ind] = np.mean(percNoPath[ind, :])
        percNoPathStd[ind] = np.mean(percNoPath[ind, :])
    return (
        pathAllMean,
        pathAllStd,
        pathOnlyPathsMean,
        pathOnlyPathsStd,
        percNoPathMean,
        percNoPathStd,
    )
