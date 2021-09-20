# import dependencies
import os

import numpy as np
from src.codebase.nodes import utils, nodes_figureS4
import yaml
from typing import Dict, Any
from matplotlib import pyplot as plt
import pickle
import time
import logging

# configure logs
logging.basicConfig(
    filename="./logs/log.txt", format="%(asctime)s - %(message)s", level=logging.INFO
)

# load project context
with open("project_context.yml", "r") as file:
    context = yaml.safe_load(file)
project_path = context["project_path"]

# load data catalog
catalog_path = os.path.join(project_path, "conf/figureS4/catalog.yml")
with open(catalog_path, "r") as file:
    CATALOG = yaml.safe_load(file)
DIR_LOAD = CATALOG["directory_load"]

# set plot parameters
plt.rcParams["figure.figsize"] = [6, 6]


def create(params) -> Dict[str, Any]:
    """Run pipeline to create figure 6

    Args:
        params ([type]): pipeline parameters stored in conf/figure6

    Returns:
        Dict[str, Any]: intermediate analysis data to for quick reproduction of figure 6
    """

    # load raw data
    file_path_load = DIR_LOAD + "A_cons_adv_50_pRand.pckl"
    A = utils.load_var(file_path_load)

    # get parameters
    REP = params["repetitions"]
    P_RAND = params["p_rnd"]

    # run analysis for different in out thresholds
    tic = time.time()
    path_lens, possible_pathsAll = nodes_figureS4.run_analysis(A, REP, P_RAND)

    # organize data
    data = {
        "path_lens": path_lens,
        "possible_pathsAll": possible_pathsAll,
        "REP": REP,
        "P_RAND": P_RAND,
    }

    # log duration
    logging.info(f"The analysis took: {time.time()-tic}")

    # plot
    plot_figure(data)
    return data


def load_data():
    """Load figure 6's intermediate data for quick reproduction

    Returns:
        Dict[str, np.ndarray]: loaded data
    """
    with open(CATALOG["intermediate_data"], "rb") as file:
        data = pickle.load(file)
    return data


def save_data(data: Dict[str, Any]):
    """Save figure 6's data for quick reproduction

    Args:
        data (Dict[str, np.ndarray]): intermediate data
            stored for quick reproduction of figure 6
    """
    with open(CATALOG["intermediate_data"], "wb") as file:
        pickle.dump(data, file)


def plot_figure(data: Dict[str, Any]):

    # get parameters
    P_RAND = data["P_RAND"]
    path_lens = data["path_lens"]
    possible_pathsAll = data["possible_pathsAll"]

    # plot
    nodes_figureS4.plot_panel_1(P_RAND, path_lens)
    nodes_figureS4.plot_panel_2(P_RAND, possible_pathsAll)
