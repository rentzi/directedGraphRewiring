import numpy as np
from src.codebase.nodes import utils
from scipy import linalg


def get_nodes_receiving_and_sending_edges(adj_matx):
    """
    Get nodes both receiving (incoming) at least one edge and sending at least one edge
    and with un number of incoming and outgoing edges below the number of nodes
    in the network

    args:

        adj_matx

    returns:

        nodes_receiving_and_sending
    """
    # number of nodes in the network
    n_nodes = adj_matx.shape[0]

    # deg[i] = number of incoming edges to the i+1 node
    degIn = np.sum(
        adj_matx > 0,
        axis=0,
        keepdims=False,
    )

    # deg[i] = number of outgoing edges from the i+1 node
    degOut = np.sum(
        adj_matx > 0,
        axis=1,
        keepdims=False,
    )

    # get nodes' with in-degrees between 0 & n_nodes (non included)
    nodes_receiving = np.where((degIn > 0) & (degIn < n_nodes - 1))

    # get nodes' with out-degrees between 0 & n_nodes (non included)
    nodes_sending = np.where((degOut > 0) & (degOut < n_nodes - 1))

    # get nodes with both indegrees and outdegrees >0 & <n_nodes
    nodes_receiving_and_sending = np.intersect1d(
        nodes_receiving[0],
        nodes_sending[0],
    )

    return nodes_receiving_and_sending


def pick_node_with_in_out_edges(
    adj_matx,
    tau,
    p_rnd_rewire,
):
    """
    Chose a node at random, keep a record of that node and other nodes

    args

        A: ?
            Adjacency matrix at initial state
        tau: ?
            heat dispersion parameter
        p_rnd_rewire: ?
            probability of random rewiring

    returns

        not_x: ?
        node_x: ?
    """

    # number of nodes in the network
    n_nodes = adj_matx.shape[0]

    # get all nodes receiving and sending edges
    nodes_receiving_and_sending = get_nodes_receiving_and_sending_edges(adj_matx)

    # If there are none, return Adjacency matrix
    if utils.check_is_null_exception(
        nodes_receiving_and_sending,
        adj_matx,
        tau,
        p_rnd_rewire,
    ):
        return adj_matx

    # else randomly pick a node
    node_x = np.random.choice(nodes_receiving_and_sending)

    # keep a record of the other nodes
    indAll = np.arange(n_nodes)
    not_x = np.delete(indAll, node_x)

    return (
        not_x,
        node_x,
    )


def compute_consensus_kernel(adj_matx, tau):
    """
    Calculate the consensus kernel

    Args:

        A:
            initial adjacency matrix
        tau
            heat dispersion parameter
    """

    # estimate the in degree Laplacian
    Din = np.diag(np.sum(adj_matx, axis=1))
    Lin = Din - adj_matx

    # calculate the consensus kernel
    kernel = linalg.expm(-tau * Lin)

    return kernel


def compute_advection_kernel(adj_matx, tau):
    """
    Calculate the advection kernel. Use Lout instead of Lin (consensus case)

    Args:

        A:
            initial adjacency matrix
        tau
            heat dispersion parameter
    """

    # estimate the out degree Laplacian
    Dout = np.diag(np.sum(adj_matx, axis=0))
    Lout = Dout - adj_matx

    # calculate the consensus kernel
    kernel = linalg.expm(-tau * Lout)

    return kernel


def generate_rand_Adj(n_nodes, edges, **kwargs):
    """
    Generate a directed random matrix

    Args:

        n_nodes:
            number of vertices
        edges:
            number of edges (nonzero entries on the random matrix)

    **kwargs

        weightDistribution:
            'binary', 'normal' or 'lognormal'
        mu:
            the mu parameter, is not valid for binary
        sig:
            the sig parameter, is not valid for binary

    Returns:

        A:
            the verticesXvertices random digraph

    """
    for (
        key,
        value,
    ) in kwargs.items():
        if key == "weightDistribution":
            weightDistribution = value
        elif key == "mu":
            mu = value
        elif key == "sig":
            sig = value

    # set constant
    EPSILON = 0.05

    # set the max number of network edges
    maxConnections = int(n_nodes * (n_nodes - 1))
    # print(
    #    "(generate_rand_Adj) maxConnections:", maxConnections,
    # )
    # print(
    #    "(generate_rand_Adj) weight distribution:", weightDistribution,
    # )

    try:

        # print("(generate_rand_Adj) Generating random adjacency matrix ...")

        # sample weights from a lognormal distribution
        if weightDistribution == "lognormal":

            if utils.not_exist("mu", "sig"):
                mu, sig = (
                    0.0,
                    1.0,
                )
                # print("(generate_rand_Adj) default mu:{}, sig:{}:".format(mu, sig,))
            randWeights = np.random.lognormal(
                mean=mu,
                sigma=sig,
                size=edges,
            )

        # ... from a normal distribution
        elif weightDistribution == "normal":

            if utils.not_exist("mu", "sig"):
                mu, sig = (
                    1.0,
                    0.25,
                )
                # print("(generate_rand_Adj) default mu:{}, sig:{}:".format(mu, sig,))
            randWeights = np.random.normal(
                loc=mu,
                scale=sig,
                size=edges,
            )
            ind = np.where(randWeights < 0)
            randWeights[ind] = EPSILON

        # ... from a binary distribution
        elif weightDistribution == "binary":

            randWeights = np.ones(edges)

        # [print info on weights for validation]
        # print(
        #    "(generate_rand_Adj) Weights generation: randWeights: type:{}, ndim: {}, shape: {}, min: {}, max: {}".format(
        #        type(randWeights),
        #        randWeights.ndim,
        #        randWeights.shape,
        #        randWeights.min(),
        #        randWeights.max(),
        #    )
        # )

        # Normalize weights such that their sum equals the number of edges
        if (weightDistribution == "normal") | (weightDistribution == "lognormal"):

            normFactor = len(randWeights) / np.sum(randWeights)
            normRandWeights = randWeights * normFactor

            # [print info on normalized weights for validation]
            # print(
            #    "(generate_rand_Adj) Weights normalization: normRandWeights: type:{}, ndim: {}, shape: {}, min: {}, max: {}".format(
            #        type(normRandWeights),
            #        normRandWeights.ndim,
            #        normRandWeights.shape,
            #        normRandWeights.min(),
            #        normRandWeights.max(),
            #    )
            # )
        else:
            normRandWeights = randWeights

        # Get the indices of 1s of a matrix the same size as A with 1s everywhere except in the diagonal
        Aones = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)
        ind = np.where(Aones)

        # Pick a random sample of those indices (# edges)
        xxRand = np.random.permutation(maxConnections)
        indRand = (
            ind[0][xxRand[:edges]],
            ind[1][xxRand[:edges]],
        )

        # build the adjacency matrix w/ those indices
        adj_matx = np.zeros((n_nodes, n_nodes))
        adj_matx[indRand] = normRandWeights
        # print(
        #    "(generate_rand_Adj) adj_matx: type:{}, ndim: {}, shape: {}, min: {}, max: {}".format(
        #        type(adj_matx),
        #        adj_matx.ndim,
        #        adj_matx.shape,
        #        adj_matx.min(),
        #        adj_matx.max(),
        #    )
        # )

    except:
        if edges > maxConnections or edges < 0:
            raise Exception("(generate_rand_Adj) Edge number out of range")
        else:
            print("error")
            return -1
    # print("(generate_rand_Adj) Completed")
    return adj_matx


def run_out_dynamics(
    AInit,
    p_rnd_rewire,
    n_rewirings,
    tau,
    flag_alg,
):
    """
    Rewires iteratively a matrix A. At each iteration the rewiring can be random
    (probability = p_rnd_rewire) or according to a consensus or advection function
    (probability = 1-p_rnd_rewire). It rewires only the OUTDEGREES. More specifically,
    during each rewiring iteration a random node k is selected and one of its outdegrees
    is cut, and a connection is added with the tail being k. It operates in the columns Works
    for both binary and weighted initial networks since this implementation just redistributes the weights

    Args:

        AInit:
            initial adjacency matrix
        p_rnd_rewire:
            probability of random rewiring
        n_rewirings:
            number of rewiring iterations
        tau:
            heat dispersion parameter
        flag_alg:
            'consensus' or 'advection' depending on which Laplacian you use for the kernel; Lin or Lout

    Returns:

        A:
            returns a rewired matrix
    """

    adj_matx = AInit.copy()
    n_nodes = adj_matx.shape[0]

    for iter_i in range(n_rewirings):
        # print('iteration is '+str(iter_i))

        (not_x, node_x,) = pick_node_with_in_out_edges(
            adj_matx,
            tau,
            p_rnd_rewire,
        )

        # find the nodes with no incomings from "x"
        x_non_head_bool = 1.0 * np.logical_not(
            adj_matx[
                not_x,
                node_x,
            ]
        )
        # print('makes its nonzeros zeros and vice versa')
        # print(x_non_head_bool)

        # rewire by network diffusion
        if np.random.random_sample() >= p_rnd_rewire:

            # calculate the consensus or advection kernel
            if flag_alg == "consensus":
                kernel = compute_consensus_kernel(adj_matx, tau)
            elif flag_alg == "advection":
                kernel = compute_advection_kernel(adj_matx, tau)

            # get node x's coldest head (excluding "x")
            x_heads = adj_matx[:, node_x].nonzero()[0]
            x_cut_head = x_heads[
                np.argmin(
                    kernel[
                        x_heads,
                        node_x,
                    ]
                )
            ]

            # Get x's hottest non-head (excluding "x")
            x_non_heads = not_x[x_non_head_bool.nonzero()[0]]
            x_wire_non_head = x_non_heads[
                np.argmax(
                    kernel[
                        x_non_heads,
                        node_x,
                    ]
                )
            ]

        else:  # else we randomly rewire

            # randomly pick one of x's head (excluding "x")
            x_heads = adj_matx[
                :,
                node_x,
            ].nonzero()[0]
            x_cut_head = np.random.choice(x_heads)

            x_wire_non_head = not_x[np.random.choice(x_non_head_bool.nonzero()[0])]
            # print("x_wire_non_head is :"+str(x_wire_non_head))

        # Warning if exception
        if x_cut_head == x_wire_non_head:
            print("PROBLEM")
            print(
                "The A nodes rewired are %d and %d with weight %f"
                % (
                    x_wire_non_head,
                    node_x,
                    adj_matx[
                        x_cut_head,
                        node_x,
                    ],
                )
            )
            print(
                "The A nodes disconnected are %d and %d"
                % (
                    x_cut_head,
                    node_x,
                )
            )

        # if rewired by network consensus: the chosen node's edge to its coldest target is
        # switched towards its hottest non target (non-tail or tail)
        # if randomly rewired: one of the chosen node's target is picked randomly and
        # replaced with a non target picked randomly
        adj_matx[x_wire_non_head, node_x,] = adj_matx[
            x_cut_head,
            node_x,
        ]
        adj_matx[
            x_cut_head,
            node_x,
        ] = 0

    return adj_matx


def run_in_dynamics(
    AInit,
    p_rnd_rewire,
    n_rewirings,
    tau,
    flag_alg,
):
    """
    Rewires iteratively a matrix A. At each iteration the rewiring can be random
    (probability = p_rnd_rewire) or according to a consensus/advection function (probability = 1-p_rnd_rewire).
    It rewires only the INDEGREES. More specifically, during each rewiring iteration a random node k is selected
    and one of its tails is cut, and a connection is added with the head being k. It operates in the rows

    Args:
        AInit:
            initial adjacency matrix
        p_rnd_rewire:
            probability of random rewiring
        n_rewirings:
            number of iterations the wiring take place
        tau:
            heat dispersion parameter
        flag_alg:
            'consensus' or 'advection' depending on which Laplacian you use for the kernel; Lin or Lout

    Returns:
        A:
            returns a rewired  matrix
    """

    adj_matx = AInit.copy()
    n_nodes = adj_matx.shape[0]

    for iter_i in range(n_rewirings):

        # randomly pick a node "x"
        (not_x, node_x,) = pick_node_with_in_out_edges(
            adj_matx,
            tau,
            p_rnd_rewire,
        )

        # Identify nodes that do not send an edge to "x
        x_non_tail_bool = 1.0 * np.logical_not(
            adj_matx[
                node_x,
                not_x,
            ]
        )

        if np.random.random_sample() >= p_rnd_rewire:  # rewire by network diffusion

            # calculate the consensus or advection kernel
            if flag_alg == "consensus":
                kernel = compute_consensus_kernel(adj_matx, tau)
            elif flag_alg == "advection":
                kernel = compute_advection_kernel(adj_matx, tau)

            # get node x's coldest tail (excluding "x")
            x_tails = adj_matx[
                node_x,
                :,
            ].nonzero()[0]
            x_cut_tail = x_tails[
                np.argmin(
                    kernel[
                        node_x,
                        x_tails,
                    ]
                )
            ]

            # get node x's hottest non-tail (excluding "x")
            x_non_tails = not_x[x_non_tail_bool.nonzero()[0]]
            x_wire_non_tail = x_non_tails[
                np.argmax(
                    kernel[
                        node_x,
                        x_non_tails,
                    ]
                )
            ]

        else:  # now we just randomly rewire

            # randomly pick one of x's tails (excluding "x")
            x_tails = adj_matx[
                node_x,
                :,
            ].nonzero()[0]
            x_cut_tail = np.random.choice(x_tails)

            # randomly pick one of x's non-tails (excluding "x")
            x_wire_non_tail = not_x[np.random.choice(x_non_tail_bool.nonzero()[0])]

        # Warning if exception
        if x_cut_tail == x_wire_non_tail:
            print("PROBLEM")
            print(
                "The A nodes rewired are %d and %d with weight %f"
                % (
                    x_wire_non_tail,
                    node_x,
                    adj_matx[node_x, x_cut_tail],
                )
            )
            print(
                "The A nodes disconnected are %d and %d"
                % (
                    x_cut_tail,
                    node_x,
                )
            )

        # if rewired by network consensus: the chosen node's edge to its coldest target is
        # switched towards its hottest non target (non-tail or tail)
        # if randomly rewired: one of the chosen node's target is picked randomly and
        # replaced with a non target picked randomly
        adj_matx[node_x, x_wire_non_tail,] = adj_matx[
            node_x,
            x_cut_tail,
        ]
        adj_matx[
            node_x,
            x_cut_tail,
        ] = 0

    return adj_matx


def run_in_and_out_dynamics(
    AInit,
    p_rnd_rewire,
    n_rewirings,
    tau,
    flag_alg,
):
    """
    Rewires iteratively a matrix A. At each iteration the rewiring can be random
    (probability= p_rnd_rewire) or according to a consensus/advection function (probability = 1-p_rnd_rewire).
    It rewires both the OUTDEGREES and INDEGREES

    Args:
        AInit:
            initial adjacency matrix
        p_rnd_rewire:
            probability of random rewiring
        n_rewirings:
            number of iterations the wiring take place
        tau:
            heat dispersion parameter
         flag_alg:
            'consensus' or 'advection' depending on which Laplacian you use for the kernel; Lin or Lout

    Returns:
        A:
            returns a rewired  matrix
    """

    adj_matx = AInit.copy()
    n_nodes = adj_matx.shape[0]

    for iter_i in range(n_rewirings):

        # pick a node at random
        (not_x, node_x,) = pick_node_with_in_out_edges(
            adj_matx,
            tau,
            p_rnd_rewire,
        )

        # find the nodes with no incomings from "x"
        x_non_head_bool = 1.0 * np.logical_not(
            adj_matx[
                not_x,
                node_x,
            ]
        )

        # Identify nodes that do not send an edge to "x
        x_non_tail_bool = 1.0 * np.logical_not(
            adj_matx[
                node_x,
                not_x,
            ]
        )

        if np.random.random_sample() >= p_rnd_rewire:  # rewire by network diffusion

            # calculate the consensus or advection kernel
            if flag_alg == "consensus":
                kernel = compute_consensus_kernel(adj_matx, tau)
            elif flag_alg == "advection":
                kernel = compute_advection_kernel(adj_matx, tau)

            # ====== REWIRING FOR THE INCOMING EDGES OF THE NODE X ====

            # get node x's coldest tail (excluding "x")
            x_tails = adj_matx[
                node_x,
                :,
            ].nonzero()[0]
            x_cut_tail = x_tails[
                np.argmin(
                    kernel[
                        node_x,
                        x_tails,
                    ]
                )
            ]

            # get node x's hottest non-tail (excluding "x")
            x_non_tails = not_x[x_non_tail_bool.nonzero()[0]]
            x_wire_non_tail = x_non_tails[
                np.argmax(
                    kernel[
                        node_x,
                        x_non_tails,
                    ]
                )
            ]

            # ====== REWIRING FOR THE OUTGOING EDGES OF THE NODE X ====

            # get node x's coldest head (excluding "x")
            x_heads = adj_matx[:, node_x].nonzero()[0]
            x_cut_head = x_heads[
                np.argmin(
                    kernel[
                        x_heads,
                        node_x,
                    ]
                )
            ]

            # Get x's hottest non-head (excluding "x")
            x_non_heads = not_x[x_non_head_bool.nonzero()[0]]
            x_wire_non_head = x_non_heads[
                np.argmax(
                    kernel[
                        x_non_heads,
                        node_x,
                    ]
                )
            ]

        else:  # or randomly rewire

            ####################################
            # randomly pick one of x's tails (excluding "x")
            x_tails = adj_matx[
                node_x,
                :,
            ].nonzero()[0]
            x_cut_tail = np.random.choice(x_tails)

            # randomly pick one of x's non-tails (excluding "x")
            x_wire_non_tail = not_x[np.random.choice(x_non_tail_bool.nonzero()[0])]

            ##################################
            # randomly pick one of x's head (excluding "x")
            x_heads = adj_matx[
                :,
                node_x,
            ].nonzero()[0]
            x_cut_head = np.random.choice(x_heads)

            x_wire_non_head = not_x[np.random.choice(x_non_head_bool.nonzero()[0])]
            # print("x_wire_non_head is :"+str(x_wire_non_head))

        if (x_cut_tail == x_wire_non_tail) | (x_cut_head == x_wire_non_head):
            print("PROBLEM")

        ##################################################################
        # Rewiring
        adj_matx[node_x, x_wire_non_tail,] = adj_matx[
            node_x,
            x_cut_tail,
        ]
        adj_matx[x_wire_non_head, node_x,] = adj_matx[
            x_cut_head,
            node_x,
        ]

        adj_matx[
            node_x,
            x_cut_tail,
        ] = 0
        adj_matx[
            x_cut_head,
            node_x,
        ] = 0
        ##################################################################

    return adj_matx


# either run advection or consensus at each iteration
def run_dynamics_advection_consensus_sequence(
    AInit,
    n_rewirings,
    tau,
):

    adj_matx = AInit.copy()
    n_nodes = adj_matx.shape[0]
    p_rnd_rewire = 0

    for iter_i in range(n_rewirings):

        # pick a node at random
        (not_x, node_x,) = pick_node_with_in_out_edges(
            adj_matx,
            tau,
            p_rnd_rewire,
        )

        # take the actual vector and make inversions 0->1 and 1->0
        # find the nodes with no incomings from "x"
        x_non_head_bool = 1.0 * np.logical_not(
            adj_matx[
                not_x,
                node_x,
            ]
        )

        # Identify nodes that do not send an edge to "x
        x_non_tail_bool = 1.0 * np.logical_not(
            adj_matx[
                node_x,
                not_x,
            ]
        )

        if (iter_i % 2) == 0:
            print("Even iteration, we do consensus")

            # ====== REWIRING FOR THE in-DEGREE OF THE NODE X ====
            kernel = compute_consensus_kernel(adj_matx, tau)

            # get node x's coldest tail (excluding "x")
            x_tails = adj_matx[
                node_x,
                :,
            ].nonzero()[0]
            x_cut_tail = x_tails[
                np.argmin(
                    kernel[
                        node_x,
                        x_tails,
                    ]
                )
            ]

            # get node x's hottest non-tail (excluding "x")
            x_non_tails = not_x[x_non_tail_bool.nonzero()[0]]
            x_wire_non_tail = x_non_tails[
                np.argmax(
                    kernel[
                        node_x,
                        x_non_tails,
                    ]
                )
            ]

            # cut and rewire in-going connections from node x
            adj_matx[node_x, x_wire_non_tail,] = adj_matx[
                node_x,
                x_cut_tail,
            ]
            adj_matx[
                node_x,
                x_cut_tail,
            ] = 0

        else:
            print("Odd iteration, we do advection")

            # ====== REWIRING FOR THE out-DEGREE OF NODE X =====

            kernel = compute_advection_kernel(adj_matx, tau)

            # get node x's coldest head (excluding "x")
            x_heads = adj_matx[:, node_x].nonzero()[0]
            x_cut_head = x_heads[
                np.argmin(
                    kernel[
                        x_heads,
                        node_x,
                    ]
                )
            ]

            # Get x's hottest non-head (excluding "x")
            x_non_heads = not_x[x_non_head_bool.nonzero()[0]]
            x_wire_non_head = x_non_heads[
                np.argmax(
                    kernel[
                        x_non_heads,
                        node_x,
                    ]
                )
            ]

            # cut and rewire out-going connections from node x
            adj_matx[x_wire_non_head, node_x,] = adj_matx[
                x_cut_head,
                node_x,
            ]
            adj_matx[
                x_cut_head,
                node_x,
            ] = 0

    return adj_matx


def run_dynamics_advection_consensus_parallel(
    AInit,
    n_rewirings,
    tau,
):

    adj_matx = AInit.copy()
    n_nodes = adj_matx.shape[0]
    p_rnd_rewire = 0

    for iter_i in range(n_rewirings):

        # pick a node at random
        (not_x, node_x,) = pick_node_with_in_out_edges(
            adj_matx,
            tau,
            p_rnd_rewire,
        )

        # take the actual vector and make inversions 0->1 and 1->0
        # find the nodes with no incomings from "x"
        x_non_head_bool = 1.0 * np.logical_not(
            adj_matx[
                not_x,
                node_x,
            ]
        )

        # Identify nodes that do not send an edge to "x
        x_non_tail_bool = 1.0 * np.logical_not(
            adj_matx[
                node_x,
                not_x,
            ]
        )

        # ====== REWIRING FOR THE in-DEGREE OF THE NODE X ====
        kernel = compute_consensus_kernel(adj_matx, tau)

        # get node x's coldest tail (excluding "x")
        x_tails = adj_matx[
            node_x,
            :,
        ].nonzero()[0]
        x_cut_tail = x_tails[
            np.argmin(
                kernel[
                    node_x,
                    x_tails,
                ]
            )
        ]

        # get node x's hottest non-tail (excluding "x")
        x_non_tails = not_x[x_non_tail_bool.nonzero()[0]]
        x_wire_non_tail = x_non_tails[
            np.argmax(
                kernel[
                    node_x,
                    x_non_tails,
                ]
            )
        ]

        # ====== REWIRING FOR THE out-DEGREE OF NODE X =====

        kernel = compute_advection_kernel(adj_matx, tau)

        # get node x's coldest head (excluding "x")
        x_heads = adj_matx[:, node_x].nonzero()[0]
        x_cut_head = x_heads[
            np.argmin(
                kernel[
                    x_heads,
                    node_x,
                ]
            )
        ]

        # Get x's hottest non-head (excluding "x")
        x_non_heads = not_x[x_non_head_bool.nonzero()[0]]
        x_wire_non_head = x_non_heads[
            np.argmax(
                kernel[
                    x_non_heads,
                    node_x,
                ]
            )
        ]

        ############################
        # cut and rewire in-going connections from node x
        adj_matx[node_x, x_wire_non_tail,] = adj_matx[
            node_x,
            x_cut_tail,
        ]
        adj_matx[
            node_x,
            x_cut_tail,
        ] = 0

        # cut and rewire out-going connections from node x
        adj_matx[x_wire_non_head, node_x,] = adj_matx[
            x_cut_head,
            node_x,
        ]
        adj_matx[
            x_cut_head,
            node_x,
        ] = 0

    return adj_matx


# either run advection or consensus depending on p_adv. For example if p_adv = 0.7, every time there is a 70% probability that advection will take place, and a 30% that consensus will take place


def run_dynamics_advection_consensus(
    AInit,
    n_rewirings,
    tau,
    p_adv,
):

    adj_matx = AInit.copy()
    n_nodes = adj_matx.shape[0]
    p_rnd_rewire = 0

    for iter_i in range(n_rewirings):

        # pick a node at random
        (not_x, node_x,) = pick_node_with_in_out_edges(
            adj_matx,
            tau,
            p_rnd_rewire,
        )

        # take the actual vector and make inversions 0->1 and 1->0
        # find the nodes with no incomings from "x"
        x_non_head_bool = 1.0 * np.logical_not(
            adj_matx[
                not_x,
                node_x,
            ]
        )

        # Identify nodes that do not send an edge to "x
        x_non_tail_bool = 1.0 * np.logical_not(
            adj_matx[
                node_x,
                not_x,
            ]
        )

        if np.random.random_sample() >= p_adv:
            # print('we do consensus')

            # ====== REWIRING FOR THE in-DEGREE OF THE NODE X ====
            kernel = compute_consensus_kernel(adj_matx, tau)

            # get node x's coldest tail (excluding "x")
            x_tails = adj_matx[
                node_x,
                :,
            ].nonzero()[0]
            x_cut_tail = x_tails[
                np.argmin(
                    kernel[
                        node_x,
                        x_tails,
                    ]
                )
            ]

            # get node x's hottest non-tail (excluding "x")
            x_non_tails = not_x[x_non_tail_bool.nonzero()[0]]
            x_wire_non_tail = x_non_tails[
                np.argmax(
                    kernel[
                        node_x,
                        x_non_tails,
                    ]
                )
            ]

            # cut and rewire in-going connections from node x
            adj_matx[node_x, x_wire_non_tail,] = adj_matx[
                node_x,
                x_cut_tail,
            ]
            adj_matx[
                node_x,
                x_cut_tail,
            ] = 0

        else:
            # print('we do advection')

            # ====== REWIRING FOR THE out-DEGREE OF NODE X =====

            kernel = compute_advection_kernel(adj_matx, tau)

            # get node x's coldest head (excluding "x")
            x_heads = adj_matx[:, node_x].nonzero()[0]
            x_cut_head = x_heads[
                np.argmin(
                    kernel[
                        x_heads,
                        node_x,
                    ]
                )
            ]

            # Get x's hottest non-head (excluding "x")
            x_non_heads = not_x[x_non_head_bool.nonzero()[0]]
            x_wire_non_head = x_non_heads[
                np.argmax(
                    kernel[
                        x_non_heads,
                        node_x,
                    ]
                )
            ]

            # cut and rewire out-going connections from node x
            adj_matx[x_wire_non_head, node_x,] = adj_matx[
                x_cut_head,
                node_x,
            ]
            adj_matx[
                x_cut_head,
                node_x,
            ] = 0

    return adj_matx


def run_dynamics_advection_consensus_with_random_rewire(
    AInit,
    n_rewirings,
    tau,
    p_adv,
    p_rnd_rewire,
):

    adj_matx = AInit.copy()
    n_nodes = adj_matx.shape[0]

    for iter_i in range(n_rewirings):

        # pick a node at random
        (not_x, node_x,) = pick_node_with_in_out_edges(
            adj_matx,
            tau,
            p_rnd_rewire,
        )

        # take the actual vector and make inversions 0->1 and 1->0
        # find the nodes with no incomings from "x"
        x_non_head_bool = 1.0 * np.logical_not(
            adj_matx[
                not_x,
                node_x,
            ]
        )

        # Identify nodes that do not send an edge to "x
        x_non_tail_bool = 1.0 * np.logical_not(
            adj_matx[
                node_x,
                not_x,
            ]
        )

        if np.random.random_sample() >= p_rnd_rewire:
            if np.random.random_sample() >= p_adv:
                # print('we do consensus')

                # ====== REWIRING FOR THE in-DEGREE OF THE NODE X ====
                kernel = compute_consensus_kernel(adj_matx, tau)

                # get node x's coldest tail (excluding "x")
                x_tails = adj_matx[
                    node_x,
                    :,
                ].nonzero()[0]
                x_cut_tail = x_tails[
                    np.argmin(
                        kernel[
                            node_x,
                            x_tails,
                        ]
                    )
                ]

                # get node x's hottest non-tail (excluding "x")
                x_non_tails = not_x[x_non_tail_bool.nonzero()[0]]
                x_wire_non_tail = x_non_tails[
                    np.argmax(
                        kernel[
                            node_x,
                            x_non_tails,
                        ]
                    )
                ]

                # cut and rewire in-going connections from node x
                adj_matx[node_x, x_wire_non_tail,] = adj_matx[
                    node_x,
                    x_cut_tail,
                ]
                adj_matx[
                    node_x,
                    x_cut_tail,
                ] = 0

            else:
                # print('we do advection')

                # ====== REWIRING FOR THE out-DEGREE OF NODE X =====

                kernel = compute_advection_kernel(adj_matx, tau)

                # get node x's coldest head (excluding "x")
                x_heads = adj_matx[:, node_x].nonzero()[0]
                x_cut_head = x_heads[
                    np.argmin(
                        kernel[
                            x_heads,
                            node_x,
                        ]
                    )
                ]

                # Get x's hottest non-head (excluding "x")
                x_non_heads = not_x[x_non_head_bool.nonzero()[0]]
                x_wire_non_head = x_non_heads[
                    np.argmax(
                        kernel[
                            x_non_heads,
                            node_x,
                        ]
                    )
                ]

                # cut and rewire out-going connections from node x
                adj_matx[x_wire_non_head, node_x,] = adj_matx[
                    x_cut_head,
                    node_x,
                ]
                adj_matx[
                    x_cut_head,
                    node_x,
                ] = 0

        else:  # or randomly rewire

            ####################################
            # randomly pick one of x's tails (excluding "x")
            x_tails = adj_matx[
                node_x,
                :,
            ].nonzero()[0]
            x_cut_tail = np.random.choice(x_tails)

            # randomly pick one of x's non-tails (excluding "x")
            x_wire_non_tail = not_x[np.random.choice(x_non_tail_bool.nonzero()[0])]

            ##################################
            # randomly pick one of x's head (excluding "x")
            x_heads = adj_matx[
                :,
                node_x,
            ].nonzero()[0]
            x_cut_head = np.random.choice(x_heads)

            x_wire_non_head = not_x[np.random.choice(x_non_head_bool.nonzero()[0])]
            # print("x_wire_non_head is :"+str(x_wire_non_head))

            if (x_cut_tail == x_wire_non_tail) | (x_cut_head == x_wire_non_head):
                print("PROBLEM")

            ##################################################################
            # Rewiring
            adj_matx[node_x, x_wire_non_tail,] = adj_matx[
                node_x,
                x_cut_tail,
            ]
            adj_matx[x_wire_non_head, node_x,] = adj_matx[
                x_cut_head,
                node_x,
            ]

            adj_matx[
                node_x,
                x_cut_tail,
            ] = 0
            adj_matx[
                x_cut_head,
                node_x,
            ] = 0
            ##################################################################

    return adj_matx
