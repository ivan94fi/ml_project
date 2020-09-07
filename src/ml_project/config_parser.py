"""Functions for options parsing."""

import argparse
import re

from tabulate import tabulate


def strtobool(val):
    """
    Convert a string representation of truth to True or False.

    Adapted from distustutils.util.strtobool.

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    if val in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError("invalid truth value {}".format(val))


def _check_torch_device(value):
    match = re.match(r"(cpu|cuda(:\d)?)$", value)
    if match is None:
        raise argparse.ArgumentTypeError(
            "Device argument '{}' cannot be parsed."
            "\n    Accepted values: ['cpu', 'cuda:n']".format(value)
        )
    return match.group(0)


def tabulate_config(config, **tabulate_kwargs):
    """Format the configuration as a table for visualization.

    Parameters
    ----------
    config : argparse.Namespace
        The configuration to parse.
    **tabulate_kwargs :
        Additional arguments passed to tabulate.

    Returns
    -------
    str
        The formatted table.

    """
    config_dict = vars(config)
    table = map(
        lambda v: (v[0], v[1] if v[1] is not None else "None"),
        sorted(config_dict.items()),
    )
    return tabulate(table, headers=("Option", "Value"), **tabulate_kwargs)


# pylint: disable=too-few-public-methods
class ParseStdRange(argparse.Action):
    """Check condition on std range passed from command line."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Perfrom the check."""
        if values[0] > values[1]:
            parser.error("lower bound should be lesser than or equal to upper bound")
        if values[0] > 100 or values[1] > 100:
            parser.error("std values should be in [0,100]")
        namespace.std_range = tuple(values)


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
        "reimplementation of Noise2Noise framework.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    main_parser.add_argument(
        "--device",
        action="store",
        default="cuda",
        type=_check_torch_device,
        help="Device where the model should be run on",
    )
    main_parser.add_argument(
        "-n",
        "--num-examples",
        default=None,
        type=int,
        metavar="N",
        help="Number of examples from the dataset to use",
    )
    main_parser.add_argument(
        "--input-size",
        default=256,
        type=int,
        help="Dimensions of the input layer of the network",
    )
    main_parser.add_argument(
        "--dont-stretch",
        action="store_const",
        const=False,
        dest="stretch",
        default=True,
        help="Retain the aspect ratio of small images when resized. If not given, "
        "small images will be stretched to match (input_size, input_size)",
    )
    main_parser.add_argument(
        "--dont-shuffle",
        action="store_const",
        const=False,
        dest="shuffle",
        default=True,
        help="Do not shuffle the dataset on loading. If not given, the dataset "
        "is shuffled randomly when loaded",
    )
    main_parser.add_argument(
        "-w",
        "--workers",
        default=4,
        type=int,
        help="Number of processes to spawn during dataset loading",
    )
    main_parser.add_argument(
        "--pin-memory",
        default=True,
        type=strtobool,
        help="Use pinned memory for tensors allocation during dataset loading. "
        "Supported values are y[es]/n[o], t[rue]/f[alse], case insensitive.",
    )

    main_parser.add_argument(
        "--dataset-root",
        type=str,
        help="The root directory of the dataset to use. This directory must "
        "directly contain the images with no subdirectories",
    )

    subparsers = main_parser.add_subparsers(
        title="Subcommands for procedures",
        description="Specify the procedure to execute",
        dest="command",
        required=True,
        help="Use '<subcommand> --help' for additional help on each subcommand",
    )
    train_parser = subparsers.add_parser(
        "train",
        help="Execute training procedure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument(
        "-e", "--epochs", default=2, type=int, help="Epochs for the training procedure"
    )
    train_parser.add_argument(
        "-b",
        "--batch-size",
        default=4,
        type=int,
        help="Minibatch size for the training procedure",
    )
    train_parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Parse configuration, define dataloaders, and exit without training",
    )
    train_parser.add_argument(
        "--train-percentage",
        default=80,
        type=lambda v: float(v) if "." in v else int(v),
        help="Percentage of the total examples to reserve for the train split. The "
        "rest is reserved for validation. The value must be in (0.0, 1.0] or (0, 100]",
    )

    train_parser.add_argument(
        "--train-mode",
        default="n2n",
        choices=["n2n", "n2c"],
        type=str,
        help="Train mode: 'n2n' to use corrupted samples and target;"
        "'n2c' to use corrupted samples and clean targets",
    )

    noise_group = train_parser.add_argument_group("Noise settings")
    noise_group.add_argument(
        "--noise-type",
        default="gaussian",
        choices=["gaussian"],
        type=str,
        help="The type of noise to add to the examples/targets",
    )

    noise_group.add_argument(
        "--std-range",
        default=(0.0, 50.0),
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        action=ParseStdRange,
        help="The standard deviation range of noise to be added",
    )

    test_parser = subparsers.add_parser(
        "test",
        help="Execute test procedure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    test_parser.add_argument("--test-opt", action="store", help="example")

    parsed_args = main_parser.parse_args(args)

    return parsed_args


if __name__ == "__main__":
    configs = parse_config()
    print(tabulate_config(configs))
