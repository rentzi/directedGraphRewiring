import os
import sys
import yaml

# get project's path
# write project config.
from src.codebase.nodes import utils

proj_path = os.getcwd()
utils.write_project_config(proj_path)

from src.codebase.pipelines import (
    figure2,
    figure3,
    figure4,
    figure5,
    figure6,
    figureS3,
    figureS4,
)


def check_run_args(run_args):
    """Check that run arguments passed to python -m main are correct"""

    # check that argument 1 exists
    try:
        flag = run_args[1]
        if flag in ["--run", "--load"]:
            try:
                pipeline_name = sys.argv[2]
            except:
                SyntaxError("Specify pipeline name. It is missing.")
        else:
            print(
                """ 
                Specify pipeline with either: 
                --run followed with a pipeline name 
                --load followed with a pipeline name 
                """
            )
    except:
        SyntaxError(
            """ 
                Specify pipeline with either:  
                --run followed with a pipeline name 
                --load followed with a pipeline name 
                """
        )
    return 0


def run_pipeline(pipeline: str):
    """Run pipeline to create specified figure

    Args:
        pipeline (str): [description]
    """

    # set parameter file
    params_file = f"conf/{pipeline}/parameters.yml"

    # load parameters
    with open(params_file, "r") as file:
        params = yaml.safe_load(file)

    # get pipeline
    pipe = eval(pipeline)

    # run pipeline
    output = pipe.create(params)

    # save intermediate data
    pipe.save_data(output)


def load(pipeline=str):
    """Load a pipeline's stored intermediate data for a quick plot

    Args:
        pipeline (str, optional): e.g., "figure2"
    """
    # load intermediate data
    data = eval(pipeline).load_data()

    # plot
    eval(pipeline).plot_figure(data)


if __name__ == "__main__":
    """Entry point"""

    # check run arguments
    check_run_args(sys.argv)

    # run pipeline
    if sys.argv[1] == "--run":
        run_pipeline(pipeline=sys.argv[2])
    elif sys.argv[1] == "--load":
        load(pipeline=sys.argv[2])

    # status
    print("(main) - Run completed")
