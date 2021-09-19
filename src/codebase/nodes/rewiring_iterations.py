import numpy as np
from src.codebase.nodes import rewiring


def run_dynamics_steps(
    n_nodes,
    edges,
    p_rnd_rewire,
    tau,
    n_rewirings_vec,
    flag_rewire_method,
    flag_alg,
    **kwargs
):
    """
    Rewires iteratively an adjacency matrix and stores the rewired adjacency matrices at the dictionary A.
    The rewirings for which it stores the adjacency matrices are indicated by n_rewirings_vec
    The method used by flag_rewire_method, the algorithm by flag_alg

    Args:
        AInit:
            initial adjacency matrix
        p_rnd_rewire:
            probability of random rewiring
        tau:
            heat dispersion parameter
        n_rewirings_vec:
            vector of the number of rewirings for which adjacency matrix is stored
        flag_rewired_method:
            'in' or 'out' or 'in_out' depending on which consensus algorithm we use
        flag_alg:
            'consensus' or 'advection' depending on which Laplacian you use for the kernel; Lin or Lout
    Returns:
        A:
            returns a dictionary with rewired adjacency matrices at different stages, i.e. A[4000] is adj. matrix after 4000 rewirings
    """

    AInit = rewiring.generate_rand_Adj(n_nodes, edges, **kwargs)
    A = {}
    A[0] = AInit
    # n_rewirings_vec =  np.insert(n_rewirings_vec, 0, 0, axis=0)

    for ind, n_rewirings in enumerate(n_rewirings_vec):

        if ind == 0:
            AInit = A[0]
            subtract = 0
        else:
            subtract = n_rewirings_vec[ind - 1]

        rewirings = n_rewirings - subtract
        if flag_rewire_method == "in_out":
            A[n_rewirings] = rewiring.run_in_and_out_dynamics(
                AInit,
                p_rnd_rewire,
                rewirings,
                tau,
                flag_alg,
            )
        elif flag_rewire_method == "in":

            A[n_rewirings] = rewiring.run_in_dynamics(
                AInit,
                p_rnd_rewire,
                rewirings,
                tau,
                flag_alg,
            )
        elif flag_rewire_method == "out":

            A[n_rewirings] = rewiring.run_out_dynamics(
                AInit,
                p_rnd_rewire,
                rewirings,
                tau,
                flag_alg,
            )

        # initialize for the next iteration
        AInit = A[n_rewirings]

    return A


def run_dynamics_diff_values(
    n_nodes,
    edges,
    p_rnd_rewire_vec,
    tau_vec,
    n_rewirings_vec,
    flag_rewire_method,
    flag_alg,
    **kwargs
):
    """
    Same as run_dynamics_steps but stores in dictionary for different tau and p_random values
    A[p_random,tau][num_rewirings] shows the adjacency matrix for p_random and tau values at num_rewirings
    The rewirings for which it stores the adjacency matrices are indicated by n_rewirings_vec
    The method used by flag_rewire_method, the algorithm by flag_alg

    Args:
        AInit:
            initial adjacency matrix
        p_rnd_rewire_vec:
            vector of the probabilities of random rewiring
        tau_vec:
            vector of the tau values
        n_rewirings_vec:
            vector of the number of rewirings for which adjacency matrix is stored
        flag_rewired_method:
            'in' or 'out' or 'in_out' depending on which consensus algorithm we use
        flag_alg:
            'consensus' or 'advection' depending on which Laplacian you use for the kernel; Lin or Lout
    Returns:
        A:
            returns a dictionary with rewired adjacency matrices
    """

    A = {}
    for p in p_rnd_rewire_vec:
        for tau in tau_vec:

            A[(p, tau)] = run_dynamics_steps(
                n_nodes,
                edges,
                p,
                tau,
                n_rewirings_vec,
                flag_rewire_method,
                flag_alg,
                **kwargs
            )

    return A


def run_dynamics_iterations(
    n_nodes,
    edges,
    p_rnd_rewire_vec,
    tau_vec,
    n_rewirings_vec,
    flag_rewire_method,
    flag_alg,
    iterations,
    **kwargs
):
    """
    Same as run_consensus_diff_values but for many iterations
    A_all[i][p_random,tau][num_rewirings] shows the i-th iteration adjacency matrix for p_random and tau values at num_rewirings
    The rewirings for which it stores the adjacency matrices are indicated by n_rewirings_vec
    The method used by flag_rewire_method

    Args:
        AInit:
            initial adjacency matrix
        p_rnd_rewire_vec:
            vector of the probabilities of random rewiring
        tau_vec:
            vector of the tau values
        n_rewirings_vec:
            vector of the number of rewirings for which adjacency matrix is stored
        flag_rewired_method:
            'in' or 'out' or 'in_out' depending on which consensus algorithm we use
        flag_alg:
            'consensus' or 'advection' depending on which Laplacian you use for the kernel; Lin or Lout
        iterations:
            number of repetitions we run the different parameters


    Returns:
        A_all:
            returns a dictionary with rewired adjacency matrices
    """
    A_all = {}
    for it in np.arange(iterations):
        print("we are at %d iteration " % (it + 1))
        A_all[it + 1] = run_dynamics_diff_values(
            n_nodes,
            edges,
            p_rnd_rewire_vec,
            tau_vec,
            n_rewirings_vec,
            flag_rewire_method,
            flag_alg,
            **kwargs
        )

    return A_all
