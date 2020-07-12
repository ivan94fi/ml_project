"""Functions for options parsing."""

import argparse
import re


def _check_torch_device(value):
    match = re.match(r"(cpu|cuda(:\d)?)$", value)
    if match is None:
        raise argparse.ArgumentTypeError(
            "Device argument '{}' cannot be parsed."
            "\n    Accepted values: ['cpu', 'cuda:n']".format(value)
        )
    return match.group(0)


def parse_config(args=None):
    """Parse the configuration for the execution.

    Parameters
    ----------
    args : list of strings
        Source of configuration. If None, `sys.argv` arguments are analyzed

    Returns
    -------
    argparse.Namespace
        The parsed arguments in a Namespace object

    """
    main_parser = argparse.ArgumentParser(
        description="Machine Learning Project: "
        "reimplementation of Noise2Noise framework."
    )
    main_parser.add_argument(
        "--device", action="store", default="cuda", type=_check_torch_device
    )

    subparsers = main_parser.add_subparsers(
        title="subcommands",
        description="specify an action to perform",
        dest="command",
        required=True,
        help="Use '<subcommand> --help' for additional help on each subcommand",
    )
    train_parser = subparsers.add_parser("train", help="Execute training procedure")
    train_parser.add_argument("--train-opt", action="store", help="example")

    test_parser = subparsers.add_parser("test", help="Execute test procedure")
    test_parser.add_argument("--test-opt", action="store", help="example")

    parsed_args = main_parser.parse_args(args)

    return parsed_args
