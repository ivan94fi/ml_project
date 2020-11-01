"""Functions for options parsing."""

import argparse
import os
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


def _check_gt_zero(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("starting epoch must be > 0")
    return ivalue


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
        "--stretch",
        default=True,
        type=strtobool,
        help="Whether to retain the aspect ratio of small images when resized. If "
        "True small images will be stretched to match (input_size, input_size); if "
        "False, small images will be upscaled uniformly in both directions until the "
        "smaller side matches input_size, then cropped to (input_size, input,size)",
    )
    main_parser.add_argument(
        "--shuffle",
        default=True,
        type=strtobool,
        help="Whether to randomly shuffle the dataset upon loading",
    )
    main_parser.add_argument(
        "--bg-generator",
        default=True,
        type=strtobool,
        help="Whether to use a background generator to retrieve data from disk",
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
        "directly contain the images with no subdirectories. As an alternative, "
        "the environment variable NOISE2NOISE_DATASET_ROOT can be use to specify "
        "the dataset root. If given, the command line option has precedence.",
    )
    main_parser.add_argument(
        "--fixed-seeds",
        default=True,
        type=strtobool,
        help="Whether to use fixed seeds for random number generators in the script",
    )
    main_parser.add_argument(
        "--seed", default=42, type=int, help="The seed for all RNGs in the program."
    )
    main_parser.add_argument(
        "--progress-bar",
        default=True,
        type=strtobool,
        help="Print a progress bar during training.",
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
        "-e",
        "--epochs",
        default=2,
        type=int,
        help="Total number of epochs for the training procedure. If a starting epoch "
        "is passed, epochs 1 to starting_epoch-1 are considered already done and are "
        "skipped",
    )
    train_parser.add_argument(
        "-b",
        "--batch-size",
        default=4,
        type=int,
        help="Minibatch size for the training procedure",
    )
    train_parser.add_argument(
        "-lr",
        "--learning-rate",
        default=0.0003,
        type=float,
        help="Initial learning rate for the optimizer",
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
    train_parser.add_argument(
        "--print-interval",
        default=20,
        type=int,
        metavar="INT",
        help="Print metrics during training every INT batches",
    )
    train_parser.add_argument(
        "--log-images",
        default=None,
        type=lambda v: int(v) if v else None,
        metavar="INT",
        help="If supplied, log the current sample, target and output images to "
        "tensorboard every INT batches during training. Pass an empty string "
        "(--log-images='') to disable",
    )
    train_parser.add_argument(
        "--log-other-metrics",
        default=300,
        type=lambda v: int(v) if v else None,
        metavar="INT",
        help="Log additionary metrics to tensorboard every INT batches during "
        "training. Pass an empty string (--log-other-metrics='') to disable",
    )

    checkpoint_group = train_parser.add_argument_group("Checkpoints settings")
    checkpoint_group.add_argument(
        "--start-from-checkpoint",
        default=None,
        type=os.path.realpath,
        metavar="PATH",
        help="Load a checkpoint file to restart train. Restart training from "
        "the epoch saved in the checkpoint",
    )
    checkpoint_group.add_argument(
        "--checkpoint-interval",
        default=10,
        type=lambda v: int(v) if v else None,
        metavar="INT",
        help="Save a checkpoint of the state of network and optimizer every INT "
        "epochs. Pass an empty string (--checkpoint-interval='') to disable",
    )
    checkpoint_group.add_argument(
        "--checkpoints-root",
        default=os.path.dirname(__file__),
        type=os.path.realpath,
        metavar="PATH",
        help="Root directory for checkpoints directory. The final save location "
        "will be PATH/checkpoints. Individual files will be saved with the "
        "following filename pattern: 'n2n_<TIMESTAMP>_<EPOCH>.pt'",
    )
    checkpoint_group.add_argument(
        "--starting-epoch",
        default=1,
        type=_check_gt_zero,
        help="Start training from this epoch (must be >= 1). Epochs in interval "
        "[1, STARTING_EPOCH-1] are skipped. Used to restart train from a checkpoint",
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
        help="The standard deviation range of noise to be added. This value must be "
        "specified as if it was applied to integer pixel values in [0, 255]. It is "
        "internally converted to be applied on floating point pixel values in [0, 1].",
    )
    noise_group.add_argument(
        "--val-std",
        default=25.0,
        type=float,
        help="The standard deviation for noise applied to validation set. Specify a "
        "value in [0, 255]",
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
