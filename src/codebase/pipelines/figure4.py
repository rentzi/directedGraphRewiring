# import dependencies
import os
from src.codebase.nodes import utils, nodes_figure4
import yaml
from typing import Dict, Any
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import pickle

# load project context
with open("project_context.yml", "r") as file:
    context = yaml.safe_load(file)
project_path = context["project_path"]

# load data catalog
catalog_path = os.path.join(project_path, "conf/figure4/catalog.yml")
with open(catalog_path, "r") as file:
    CATALOG = yaml.safe_load(file)
DIR_LOAD = CATALOG["directory_load"]

# set plot parameters
plt.rcParams["figure.figsize"] = [6, 6]


def create(params) -> Dict[str, Any]:
    """Run pipeline to create figure 4

    Args:
        params ([type]): pipeline parameters stored in conf/figure4

    Returns:
        Dict[str, Any]: intermediate analysis data to for quick reproduction of figure 4
    """
    # get analysis' parameters
    MIN_CON = params["min_con"]
    MAX_CON = params["max_con"]
    P_RAND = params["pRand"]
    REP = params["repetitions"]
    DEG_SEL = np.array(params["degSel"])

    # load raw data
    file_path_load = DIR_LOAD + "A_cons_adv_50_pRand.pckl"
    A = utils.load_var(file_path_load)

    # run analysis
    path_lens, possible_pathsAll, connections = nodes_figure4.run_analysis(
        MIN_CON, MAX_CON, P_RAND, REP, A
    )

    # get metrics' figure parameters
    MIN_CON = params["figures"]["min_con"]
    MAX_CON = params["figures"]["max_con"]
    MS = params["figures"]["ms"]
    LW = params["figures"]["LW"]
    colorsPlot = params["figures"]["colorsPlot"]
    shapePointNoLine = params["figures"]["shapePointNoLine"]

    # organize data
    data = {
        "path_lens": path_lens,
        "possible_pathsAll": possible_pathsAll,
        "connections": connections,
        "MIN_CON": MIN_CON,
        "MAX_CON": MAX_CON,
        "P_RAND": P_RAND,
        "DEG_SEL": DEG_SEL,
        "MS": MS,
        "LW": LW,
        "COLOR_PLOT": colorsPlot,
        "SHAPE_POINT_NO_LINE": shapePointNoLine,
    }
    # plot
    plot_figure(data)
    return data


def load_data():
    """Load figure 4's intermediate data for quick reproduction

    Returns:
        Dict[str, np.ndarray]: loaded data
    """
    with open(CATALOG["intermediate_data"], "rb") as file:
        data = pickle.load(file)
    return data


def save_data(data: Dict[str, Any]):
    """Save figure 4's data for quick reproduction

    Args:
        data (Dict[str, np.ndarray]): intermediate data
            stored for quick reproduction of figure 4
    """
    with open(CATALOG["intermediate_data"], "wb") as file:
        pickle.dump(data, file)


def plot_figure(data: Dict[str, Any]):
    """Plot figure 4

    Args:
        data (Dict[str, Any]): intermediate data for
             quick reproduction of figure 4
    """
    # get parameters
    MIN_CON = data["MIN_CON"]
    MAX_CON = data["MAX_CON"]
    P_RAND = data["P_RAND"]
    DEG_SEL = data["DEG_SEL"]
    MS = data["MS"]
    LW = data["LW"]
    COLOR_PLOT = data["COLOR_PLOT"]
    SHAPE_POINT_NO_LINE = data["SHAPE_POINT_NO_LINE"]

    # get data
    path_lens = data["path_lens"]
    possible_pathsAll = data["possible_pathsAll"]
    connections = data["connections"]

    # plot figure 4A
    degAll = nodes_figure4.plot_fig4A(
        MIN_CON,
        MAX_CON,
        P_RAND,
        DEG_SEL,
        connections,
        MS,
        LW,
        COLOR_PLOT,
        SHAPE_POINT_NO_LINE,
    )

    # plot figure 4B
    nodes_figure4.plot_fig4B(
        P_RAND,
        DEG_SEL,
        path_lens,
        connections,
        MS,
        LW,
        COLOR_PLOT,
        SHAPE_POINT_NO_LINE,
        degAll,
    )

    # plot figure 4C
    nodes_figure4.plot_fig4C(
        P_RAND,
        DEG_SEL,
        possible_pathsAll,
        connections,
        MS,
        LW,
        COLOR_PLOT,
        SHAPE_POINT_NO_LINE,
        degAll,
    )
