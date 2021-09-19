# import dependencies
import os
from src.codebase.nodes import utils, nodes_figure5
import yaml
from typing import Dict, Any
from matplotlib import pyplot as plt
import pickle

# load project context
with open("project_context.yml", "r") as file:
    context = yaml.safe_load(file)
project_path = context["project_path"]

# load data catalog
catalog_path = os.path.join(project_path, "conf/figure5/catalog.yml")
with open(catalog_path, "r") as file:
    CATALOG = yaml.safe_load(file)
DIR_LOAD = CATALOG["directory_load"]

# set plot parameters
plt.rcParams["figure.figsize"] = [6, 6]


def create(params) -> Dict[str, Any]:
    """Run pipeline to create figure 5

    Args:
        params ([type]): pipeline parameters stored in conf/figure5

    Returns:
        Dict[str, Any]: intermediate analysis data to for quick reproduction of figure 5
    """

    # get parameters
    P_RAND = params["pRand"]
    REP = params["repetitions"]
    IN_OUT_THRESH = params["inOutThresh"]
    P_R = params["pR"]
    FLAG = "_cons_adv_50_"

    # load raw data
    file_path_load = DIR_LOAD + "A" + FLAG + "pRand.pckl"
    A = utils.load_var(file_path_load)

    # run analysis
    (
        inOutDict,
        numConnectOut,
        numNoConnectOut,
        numConnectIn,
        numNoConnectIn,
        nodesAverage,
    ) = nodes_figure5.run_analysis(P_RAND, REP, IN_OUT_THRESH, P_R, FLAG, A)

    # organize data
    data = {
        "numConnectOut": numConnectOut,
        "numNoConnectOut": numNoConnectOut,
        "numConnectIn": numConnectIn,
        "numNoConnectIn": numNoConnectIn,
        "nodesAverage": nodesAverage,
        "inOutDict": inOutDict,
        "IN_OUT_THRESH": IN_OUT_THRESH,
        "P_R": P_R,
        "FLAG": FLAG,
        "REP": REP,
    }

    # plot
    plot_figure(data)
    return data


def load_data():
    """Load figure 5's intermediate data for quick reproduction

    Returns:
        Dict[str, np.ndarray]: loaded data
    """
    with open(CATALOG["intermediate_data"], "rb") as file:
        data = pickle.load(file)
    return data


def save_data(data: Dict[str, Any]):
    """Save figure 5's data for quick reproduction

    Args:
        data (Dict[str, np.ndarray]): intermediate data
            stored for quick reproduction of figure 5
    """
    with open(CATALOG["intermediate_data"], "wb") as file:
        pickle.dump(data, file)


def plot_figure(data: Dict[str, Any]):

    # get parameters
    IN_OUT_THRESH = data["IN_OUT_THRESH"]
    P_R = data["P_R"]
    REP = data["REP"]
    FLAG = data["FLAG"]
    numConnectOut = data["numConnectOut"]
    numNoConnectOut = data["numNoConnectOut"]
    numConnectIn = data["numConnectIn"]
    numNoConnectIn = data["numNoConnectIn"]
    nodesAverage = data["nodesAverage"]
    inOutDict = data["inOutDict"]

    # plot panel 1
    nodes_figure5.plot_panel_1(
        IN_OUT_THRESH, P_R, numConnectOut, numNoConnectOut, numConnectIn, numNoConnectIn
    )

    # plot panel 2
    nodes_figure5.plot_panel_2(IN_OUT_THRESH, P_R, nodesAverage)
