# import dependencies
import os
from src.codebase.nodes import utils, nodes_figure6
import yaml
from typing import Dict, Any
from matplotlib import pyplot as plt
import pickle
import seaborn as sns

# load project context
with open("project_context.yml", "r") as file:
    context = yaml.safe_load(file)
project_path = context["project_path"]

# load data catalog
catalog_path = os.path.join(project_path, "conf/figure6/catalog.yml")
with open(catalog_path, "r") as file:
    CATALOG = yaml.safe_load(file)
DIR_LOAD = CATALOG["directory_load"]

# set plot parameters
plt.rcParams["figure.figsize"] = [6, 6]


def create(params):
    """Run pipeline to create figure 6

    Args:
        params ([type]): pipeline parameters stored in conf/figure6

    Returns:
        Dict[str, Any]: intermediate analysis data to for quick reproduction of figure 6
    """

    # get parameters
    FLAG = CATALOG["flag"]
    P_RAND = params["pRand"]
    REP = params["repetitions"]
    pR = params["pR"]
    IN_OUT_THRESH = params["inOutThresh"]
    NUM_NODES = params["numNodes"]
    MARKERS = params["MARKERS"]
    COLORS = params["COLORS"]
    NUM_ITER = params["numIterations"]
    P_RAND_SAMPLE = params["P_RAND_SAMPLE"]
    DENSITY_TYPE = params["densityType"]

    # load raw data
    file_path_load = DIR_LOAD + "A" + FLAG + "pRand.pckl"
    A = utils.load_var(file_path_load)

    # run analysis
    nodesIntermed, densityIntermed, densitySimple = nodes_figure6.run_analysis(
        FLAG, P_RAND, REP, pR, IN_OUT_THRESH, NUM_NODES, A
    )

    # organize data
    data = {
        "densityIntermed": densityIntermed,
        "densitySimple": densitySimple,
        "nodesIntermed": nodesIntermed,
        "pR": pR,
        "IN_OUT_THRESH": IN_OUT_THRESH,
        "MARKERS": MARKERS,
        "COLORS": COLORS,
        "NUM_ITER": NUM_ITER,
        "P_RAND_SAMPLE": P_RAND_SAMPLE,
        "DENSITY_TYPE": DENSITY_TYPE,
    }

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

    # get data and parameters
    densityIntermed = data["densityIntermed"]
    densitySimple = data["densitySimple"]
    nodesIntermed = data["nodesIntermed"]
    pR = data["pR"]
    IN_OUT_THRESH = data["IN_OUT_THRESH"]
    MARKERS = data["MARKERS"]
    COLORS = data["COLORS"]
    NUM_ITER = data["NUM_ITER"]
    P_RAND_SAMPLE = data["P_RAND_SAMPLE"]
    DENSITY_TYPE = data["DENSITY_TYPE"]

    # panel 1
    nodes_figure6.plot_panel_1(
        pR, IN_OUT_THRESH, MARKERS, COLORS, densityIntermed, densitySimple
    )

    # panel 2
    df = nodes_figure6.plot_panel_2(
        NUM_ITER, P_RAND_SAMPLE, DENSITY_TYPE, densityIntermed, densitySimple
    )

    # panel 3
    sns.boxplot(x="type", y="densities", hue="pRand", data=df)
    plt.show()

    from ipdb import set_trace

    set_trace()

    # panel 4 [TODO]: debug
    nodes_figure6.plot_panel_3(df)

    # panel 5
    nodes_figure6.plot_panel_4(pR, IN_OUT_THRESH, nodesIntermed)
