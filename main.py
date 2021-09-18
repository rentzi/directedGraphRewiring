import os
import sys
import yaml
from src.codebase.pipelines import figure2


def check_run_args(run_args):
    """Check that the run arguments are correct"""
    try:
        flag = run_args[1]
        if flag in ["--run", "--load"]:
            try:
                pipeline_name = sys.argv[2]
            except:
                SyntaxError("Pipeline name is missing")
        else:
            print(
                """ 
                Define 
                --run followed with a pipeline name 
                --load followed with a pipeline name 
                """
            )
    except:
        SyntaxError(
            """ 
                Define 
                --run followed with a pipeline name 
                --load followed with a pipeline name 
                """
        )
    return 0


def write_project_config():
    """Write project configuration"""
    with open("project_context.yml", "w") as file:
        yaml.dump({"project_path": proj_path}, file)


def run_pipeline(pipeline: "figure2"):

    # get parameters file
    params_file = f"conf/{pipeline}/parameters.yml"

    # load parameters and run pipeline
    with open(params_file, "r") as file:
        params = yaml.safe_load(file)
    output = figure2.create(params)
    figure2.save_data(output)


def load(pipeline="figure2"):
    tmp = eval(pipeline).load_data()
    eval(pipeline).plot_digraphs(tmp)


if __name__ == "__main__":
    """Entry point"""

    # get project's path
    proj_path = os.getcwd()

    # check run arguments
    check_run_args(sys.argv)

    # store in project's config file
    write_project_config()

    # run pipeline
    if sys.argv[1] == "--run":
        run_pipeline(pipeline=sys.argv[2])
    elif sys.argv[1] == "--load":
        load(pipeline=sys.argv[2])

    # status
    print("(main) - Run completed")
