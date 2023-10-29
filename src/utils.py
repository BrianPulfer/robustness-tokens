import os
import warnings
from argparse import ArgumentParser

import yaml


def read_config():
    """Reads the yaml configuration file passed as argument and returns a dictionary with the configuration parameters."""
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
