import os
import sys
import yaml
from src.codebase.pipelines import figure2, figure3, figure4


def check_run_args(run_args):
    """Check that run arguments passed to python -m main are correct"""
    # check that arg 1 exists
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


def write_project_config():
    """Write project configuration"""
    with open("project_context.yml", "w") as file:
        yaml.dump({"project_path": proj_path}, file)


def run_pipeline(pipeline: str):
    """Run a pipeline

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
    """Load a pipeline intermediate data and plot figure

    Args:
        pipeline (str, optional): e.g.,"figure2"
    """
    # load intermediate data
    data = eval(pipeline).load_data()

    # plot
    eval(pipeline).plot_figure(data)


if __name__ == "__main__":
    """Entry point"""

    # get project's path
    proj_path = os.getcwd()

    # check run arguments
    check_run_args(sys.argv)

    # write project config.
    write_project_config()

    # run pipeline
    if sys.argv[1] == "--run":
        run_pipeline(pipeline=sys.argv[2])
    elif sys.argv[1] == "--load":
        load(pipeline=sys.argv[2])

    # status
    print("(main) - Run completed")
