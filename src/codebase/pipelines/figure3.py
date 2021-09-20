# Import dependencies
import os
import yaml
from typing import Dict, Any
import pickle
import matplotlib.pyplot as plt

# import custom functions
from src.codebase.nodes import utils, nodes_figure3

# load project context
with open("project_context.yml", "r") as file:
    context = yaml.safe_load(file)
project_path = context["project_path"]

# load data catalog
catalog_path = os.path.join(project_path, "conf/figure3/catalog.yml")
with open(catalog_path, "r") as file:
    CATALOG = yaml.safe_load(file)
FLAG = CATALOG["flag"]
DIR_LOAD = CATALOG["directory_load"]
FILE_LOAD_PATH = DIR_LOAD + "A" + FLAG + "pRand.pckl"


def create(params) -> Dict[str, Any]:
    """Run pipeline to create figure 3

    Args:
        params ([type]): [description]

    Returns:
        Dict[str, Any]: [description]
    """
    # get parameters
    P_R = params["metrics"]["pR"]
    REP = params["metrics"]["repetitions"]
    P_R_AGGREGATE = params["aggregate"]["pR"]
    REP_AGGREGATE = params["aggregate"]["repetitions"]

    # Calculate metrics
    A = utils.load_var(FILE_LOAD_PATH)
    pathDict, cycleDict = nodes_figure3.calculate_graph_metrics(P_R, REP, FLAG, A)
    data = {
        "path": pathDict,
        "cycle": cycleDict,
        "P_R_AGGREGATE": P_R_AGGREGATE,
        "REP_AGGREGATE": REP_AGGREGATE,
    }
    # plot
    plot_figure(data)
    return data


def load_data():
    """Load figure 3's intermediate data for quick reproduction

    Returns:
        Dict[str, np.ndarray]: loaded data
    """
    with open(CATALOG["path_data"], "rb") as file:
        path_data = pickle.load(file)
    with open(CATALOG["cycle_data"], "rb") as file:
        cycle_data = pickle.load(file)
    with open(CATALOG["P_R_AGGREGATE"], "rb") as file:
        P_R_AGGREGATE = pickle.load(file)
    with open(CATALOG["REP_AGGREGATE"], "rb") as file:
        REP_AGGREGATE = pickle.load(file)
    return {
        "path": path_data,
        "cycle": cycle_data,
        "P_R_AGGREGATE": P_R_AGGREGATE,
        "REP_AGGREGATE": REP_AGGREGATE,
    }


def save_data(data: Dict[str, Any]):
    """Save metrics data

    Args:
        data (Dict[str, Any]):
            keys:
                "path" : [description]
                "cycle": [description]
    """
    # get metrics data
    pathDict = data["path"]
    cycleDict = data["cycle"]
    P_R_AGGREGATE = data["P_R_AGGREGATE"]
    REP_AGGREGATE = data["REP_AGGREGATE"]

    # set write paths
    directory_save = "data/02_intermediate/figure3/pathMetrics/"
    path_file_path = directory_save + "pathDict.pickle"
    cycle_file_path = directory_save + "cycleDict.pickle"
    P_R_AGGREGATE_file_path = directory_save + "P_R_AGGREGATE.pickle"
    REP_AGGREGATE_file_path = directory_save + "REP_AGGREGATE.pickle"

    # write
    utils.save_var(pathDict, path_file_path)
    utils.save_var(cycleDict, cycle_file_path)
    utils.save_var(P_R_AGGREGATE, P_R_AGGREGATE_file_path)
    utils.save_var(REP_AGGREGATE, REP_AGGREGATE_file_path)


def plot_figure(data):
    """Plot metrics

    Args:
        pR ([type]): [description]
        pathAllMean ([type]): [description]
        pathAllStd ([type]): [description]
        pathOnlyPathsMean ([type]): [description]
        pathOnlyPathsStd ([type]): [description]
        percNoPathMean ([type]): [description]
        percNoPathStd ([type]): [description]
    """
    # get data and parameters
    pathDict = data["path"]
    P_R_AGGREGATE = data["P_R_AGGREGATE"]
    REP_AGGREGATE = data["REP_AGGREGATE"]

    # Calculate metrics
    (
        pathAllMean,
        pathAllStd,
        pathOnlyPathsMean,
        pathOnlyPathsStd,
        percNoPathMean,
        percNoPathStd,
    ) = nodes_figure3.calculate_aggregate_metrics(
        P_R_AGGREGATE, REP_AGGREGATE, FLAG, pathDict
    )

    plt.plot(P_R_AGGREGATE, pathOnlyPathsMean, color="blue", linewidth=2)
    plt.errorbar(
        P_R_AGGREGATE,
        pathOnlyPathsMean,
        pathOnlyPathsStd.squeeze(),
        marker="o",
        color="blue",
        linewidth=2,
        label="Only paths",
    )

    # plot path all mean
    plt.plot(P_R_AGGREGATE, pathAllMean, color="cyan", linewidth=2)
    plt.errorbar(
        P_R_AGGREGATE,
        pathAllMean,
        pathAllStd.squeeze(),
        marker="v",
        color="cyan",
        linewidth=2,
        label="All combinations",
    )
    plt.legend()
    plt.xticks(P_R_AGGREGATE)
    plt.yticks([0, 1, 2, 3, 4, 5])
    plt.ylim([0, 6])
    plt.xlabel("Prandom")
    plt.ylabel("Path length")
    plt.show()

    # plot 2
    plt.plot(P_R_AGGREGATE, percNoPathMean, color="red", linewidth=2)
    plt.errorbar(
        P_R_AGGREGATE,
        percNoPathMean,
        percNoPathStd.squeeze(),
        marker=">",
        color="red",
        linewidth=2,
        label="% no path",
    )
    plt.xticks(P_R_AGGREGATE)
    plt.xlabel("Prandom")
    plt.ylabel("Percentage of pairs with no path")
    plt.show()
