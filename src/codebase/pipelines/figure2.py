# import dependencies
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import pickle

# load project context
with open("project_context.yml", "r") as file:
    context = yaml.safe_load(file)
project_path = context["project_path"]

# load data catalog
catalog_path = os.path.join(project_path, "conf/figure2/catalog.yml")
with open(catalog_path, "r") as file:
    CATALOG = yaml.safe_load(file)

# import custom modules
from src.codebase.nodes import rewiring_iterations

# plot parameters
plt.rcParams["figure.figsize"] = [20, 10]


def plot_digraphs(data):
    """Plot directed graphs for different rewiring schemes

    Args:
        data ([type]): Data produced by .create() function
    """
    for scheme in data:
        for ind, key in enumerate(data[scheme].keys()):
            ttl = str(key) + " rewirings"
            plt.subplot(1, 6, ind + 1)
            plt.title(ttl)
            plt.imshow(data[scheme][key], cmap="Greys")
        plt.show()


def save_data(intermediate_data: Dict[str, np.ndarray]):
    """Save figure 2's data for quick reproduction

    Args:
        intermediate_data (Dict[str, np.ndarray]): [description]
    """
    # write intermediate result in path
    with open(CATALOG["intermediate_data"], "wb") as file:
        pickle.dump(intermediate_data, file)


def load_data():
    """Load figure 2's intermediate data for quick reproduction

    Returns:
        Dict[str, np.ndarray]: loaded data
    """
    with open(CATALOG["intermediate_data"], "rb") as file:
        data = pickle.load(file)
    return data


def create(
    params: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Create figure 2

    Args:
        params (dict): [description]

    Returns:
        Dict[np.ndarray, np.ndarray]: Dictionary holding information for two analyses "AConsensusIn" and "AAdvectionOut"

            ```python
            outs["AConsensusIn"].keys()
            # 0, 100, 200, 400, 800, 1600
            outs["AAdvectionOut"].keys()
            # 0, 100, 200, 400, 800, 1600
            ```
    """
    # get parameters
    SCHEME = params["SCHEME"]
    N_VERTEX = params["N_VERTEX"]
    P_RAND = params["P_RAND"]
    TAU = params["TAU"]
    N_REWIRING_VEC = params["N_REWIRING_VEC"]
    FLAG_REWIRE_METHOD = params["FLAG_REWIRE_METHOD"]
    FLAG_ALG = params["FLAG_ALG"]
    edge = int(np.round(2 * np.log(N_VERTEX) * (N_VERTEX - 1), decimals=0))

    # run analyses
    outs = dict()
    for ix, analysis in enumerate(SCHEME):
        print("(figure2.create) Running analysis:", analysis)
        out = rewiring_iterations.run_dynamics_steps(
            N_VERTEX,
            edge,
            P_RAND,
            TAU,
            N_REWIRING_VEC,
            flag_rewire_method=FLAG_REWIRE_METHOD[ix],
            flag_alg=FLAG_ALG[ix],
            weightDistribution="binary",
        )

        # store graphs
        outs[SCHEME[ix]] = out

    # plot graphs
    plot_digraphs(outs)
    return outs
