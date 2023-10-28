import os
import yaml
import warnings
from argparse import ArgumentParser


def read_config():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config file")

    args = vars(parser.parse_args())

    if args["config"] is not None:
        if os.path.isfile(args["config"]):
            with open(args["config"], "r") as f:
                args = yaml.safe_load(f)
        else:
            warnings.warn(f"Config file {args['config']} not found.")
            exit()
    print("\n\nProgram arguments:\n", args, "\n\n")
    return args
