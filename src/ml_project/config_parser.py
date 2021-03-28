"""Functions for options parsing."""
# pylint: disable=C0103,R0903,R0912

import argparse
import json
import os
import re
import shutil
import warnings
from datetime import datetime

from tabulate import tabulate

from ml_project.datasets import (
    ENV_VAR_DATASET_ROOT,
    ENV_VAR_TEST_DATASET_ROOT,
    ENV_VAR_VALIDATION_DATASET_ROOT,
    read_directory_from_env,
)
from ml_project.utils import normalize_path

# This timestamp will be used as name for the current experiment.
# Specifically, the object that will have this name will be:
#     - the tensorboard record file (contained in runs directory)
#     - the directory containing model checkpoints and config in results
TIMESTAMP_FORMAT = "%b%d_%H-%M-%S"
TIMESTAMP = datetime.now().strftime(TIMESTAMP_FORMAT)


class directory_structure:
    """Settings for the root directoty.

    Directory structure:
    root (default: this file's directory)
        results
            <TIMESTAMP1>
                checkpoints
                run_config.json
                <tb_events_file>
            <TIMESTAMP2>
            ...
        test_runs
            test_<TRAIN_TIMESTAMP1>
                <TIMESTAMP1>
                    events
                    train_config.json
                <TIMESTAMP1>
                    events
                    train_config.json
            ...

    """

    # The root of all produced artifacts: defaults to this file's directory.
    # results directory and runs directory will be created here
    ROOT_DIR = None

    # Path where experiment related artifacts will be placed
    RESULTS_PATH = None

    # The current experiment path
    CURRENT_EXP_PATH = None

    # The checkpoints and config path for the current experiment
    CHECKPOINTS_PATH = None
    RUN_CONFIG_PATH = None

    # Test tensorboard runs
    TEST_RUNS_PATH = None

    @classmethod
    def update(cls, _root_dir):
        """Update the global variables with the passed root_dir."""
        cls.ROOT_DIR = normalize_path(_root_dir)

        cls.RESULTS_PATH = os.path.join(cls.ROOT_DIR, "results")

        cls.CURRENT_EXP_PATH = os.path.join(cls.RESULTS_PATH, TIMESTAMP)

        cls.CHECKPOINTS_PATH = os.path.join(cls.CURRENT_EXP_PATH, "checkpoints")
        cls.RUN_CONFIG_PATH = os.path.join(cls.CURRENT_EXP_PATH, "run_config.json")

        cls.TEST_RUNS_PATH = os.path.join(cls.ROOT_DIR, "test_runs")

    @classmethod
    def create(cls):
        """Create the directory tree."""
        if not os.path.isdir(cls.ROOT_DIR):
            os.mkdir(cls.ROOT_DIR)
        if not os.path.isdir(cls.RESULTS_PATH):
            os.mkdir(cls.RESULTS_PATH)
        if not os.path.isdir(cls.CURRENT_EXP_PATH):
            os.mkdir(cls.CURRENT_EXP_PATH)
        if not os.path.isdir(cls.CHECKPOINTS_PATH):
            os.mkdir(cls.CHECKPOINTS_PATH)
        if not os.path.isdir(cls.TEST_RUNS_PATH):
            os.mkdir(cls.TEST_RUNS_PATH)


DEFAULT_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
directory_structure.update(DEFAULT_ROOT_DIR)


def strtobool(val):
    """
    Convert a string representation of truth to True or False.

    Adapted from distutils.util.strtobool.

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


class ParseNoiseParamRange(argparse.Action):
    """Parse the range of noise parameters."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Perfrom the check."""
        if values[0] > values[1]:
            parser.error("lower bound should be lesser than or equal to upper bound")
        namespace.train_params = tuple(values)


def define_parser():
    """Define the command line options as a parser object."""
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
        "--root-dir",
        default=directory_structure.ROOT_DIR,
        type=normalize_path,
        help="The root directory where artifacts will be created. This directory "
        "will contain the runs directory for tensorboard and the results directory "
        "with the checkpoints produced during training.",
    )
    main_parser.add_argument(
        "--progress-bar",
        default=True,
        type=strtobool,
        help="Print a progress bar during training.",
    )
    main_parser.add_argument(
        "--print-interval",
        default=20,
        type=int,
        metavar="INT",
        help="Print metrics during training every INT batches",
    )
    main_parser.add_argument(
        "--log-images",
        default=None,
        type=lambda v: int(v) if v else None,
        metavar="INT",
        help="If supplied, log the current sample, target and output images to "
        "tensorboard every INT batches during training. Pass an empty string "
        "(--log-images='') to disable",
    )
    main_parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Parse configuration, define dataloaders, and exit without training",
    )

    noise_group = main_parser.add_argument_group("Noise settings")
    noise_group.add_argument(
        "--noise-type",
        default="gaussian",
        choices=["gaussian", "poisson", "textual", "random_inpulse"],
        type=str,
        help="The type of noise to add to the examples/targets",
    )
    noise_group.add_argument(
        "--brown-gaussian-std",
        default=None,
        type=float,
        metavar="STD",
        help="Use a brown gaussian noise, with the specified standard deviation",
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
        "-n",
        "--num-examples",
        default=None,
        type=int,
        metavar="N",
        help="Number of examples from the dataset to use",
    )
    train_parser.add_argument(
        "--input-size",
        default=256,
        type=int,
        help="Dimensions of the input layer of the network",
    )
    train_parser.add_argument(
        "--stretch",
        default=True,
        type=strtobool,
        help="Whether to retain the aspect ratio of small images when resized. If "
        "True small images will be stretched to match (input_size, input_size); if "
        "False, small images will be upscaled uniformly in both directions until the "
        "smaller side matches input_size, then cropped to (input_size, input,size)",
    )
    train_parser.add_argument(
        "--shuffle",
        default=True,
        type=strtobool,
        help="Whether to randomly shuffle the dataset upon loading",
    )
    train_parser.add_argument(
        "--fixed-seeds",
        default=False,
        type=strtobool,
        help="Whether to use fixed seeds for random number generators in the script",
    )
    train_parser.add_argument(
        "--seed", default=42, type=int, help="The seed for all RNGs in the program."
    )
    train_parser.add_argument(
        "-e",
        "--epochs",
        default=2,
        type=int,
        help="Total number of epochs for the training procedure.",
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
        "--lr-dampen-percentage",
        default=60,
        type=int,
        metavar="PERCENTAGE",
        help="Execute learning rate dampening in the last PERCENTAGE of epochs",
    )
    train_parser.add_argument(
        "--lr-scheduling-method",
        default="cosine",
        choices=["cosine", "reduce_on_plateau"],
        type=str,
        metavar="METHOD",
        help="The learning rate scheduling method to use during training.",
    )
    train_parser.add_argument(
        "--use-external-validation",
        default=False,
        type=strtobool,
        help="Whether to use a different dataset from the training one for validation.",
    )
    train_parser.add_argument(
        "--dataset-root",
        type=str,
        help="The root directory of the dataset to use. This directory must "
        "directly contain the images with no subdirectories. As an alternative, "
        "the environment variable NOISE2NOISE_DATASET_ROOT can be used to specify "
        "the dataset root. If given, the command line option has precedence.",
    )
    train_parser.add_argument(
        "--val-dataset-root",
        type=str,
        help="The root directory of the validation dataset to use. This directory must "
        "directly contain the images with no subdirectories. As an alternative, "
        "the environment variable NOISE2NOISE_VALIDATION_DATASET_ROOT can be used "
        "to specify the dataset root. If given, the command line option has "
        "precedence.",
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
        "--log-other-metrics",
        default=300,
        type=lambda v: int(v) if v else None,
        metavar="INT",
        help="Log additionary metrics to tensorboard every INT batches during "
        "training. Pass an empty string (--log-other-metrics='') to disable",
    )

    checkpoint_group = train_parser.add_argument_group(
        "Checkpoints settings",
        description="Checkpoints will be created in "
        "<ROOT_DIR>/results/<TIMESTAMP>/checkpoints (only <ROOT_DIR> is configurable)."
        "\nThe individual checkpoint files will follow this naming convention: "
        "'n2n_<TIMESTAMP>_<epoch>.pt'.",
    )
    checkpoint_group.add_argument(
        "--start-from-checkpoint",
        default=None,
        type=normalize_path,
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

    noise_group = train_parser.add_argument_group("Noise settings")
    noise_group.add_argument(
        "--train-params",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        action=ParseNoiseParamRange,
        help="The range of the noise parameter values for training. For gaussian "
        "noise it is the standard deviation, for Poisson it is the rate. NOTE: "
        "the scale for gaussian noise std is [0, 255], then internally converted to "
        "[0, 1]." + default_noise_params_help("train"),
    )
    noise_group.add_argument(
        "--val-param",
        type=float,
        help="The noise parameter for validation. For gaussian noise, specify a "
        "value in [0, 255]." + default_noise_params_help("val"),
    )

    test_parser = subparsers.add_parser(
        "test",
        help="Execute test procedure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    checkpoint_group = test_parser.add_argument_group("Checkpoints settings")
    checkpoint_group.add_argument(
        "test_checkpoint",
        type=normalize_path,
        metavar="CHECKPOINT_PATH",
        help="Load a checkpoint file to test.",
    )

    test_parser.add_argument(
        "--test-dataset-root",
        type=str,
        help="The root directory of the test dataset to use. This directory must "
        "directly contain the images with no subdirectories. As an alternative, "
        "the environment variable NOISE2NOISE_TEST_DATASET_ROOT can be used to "
        "specify the dataset root. If given, the command line option has precedence.",
    )

    noise_group = test_parser.add_argument_group("Noise settings")
    noise_group.add_argument(
        "--test-param",
        type=float,
        help="The noise parameter for validation. For gaussian noise, specify a "
        "value in [0, 255]." + default_noise_params_help("test"),
    )

    return main_parser


default_noise_params = {
    "gaussian": {"train": (0.0, 50), "val": 25, "test": 25},
    "poisson": {"train": (0.0, 50), "val": 30, "test": 30},
    "textual": {"train": (0.0, 0.5), "val": 0.25, "test": 0.25},
    "random_inpulse": {"train": (0.0, 0.95), "val": 0.7, "test": 0.7},
}


def default_noise_params_help(phase):
    """Generate help text with default values for noise params."""
    info_str = " If not given, the following defaults are used: "
    # params_dict = _default_noise_params_t[phase]
    params_gen = (
        (k.replace("_", " ").capitalize(), v[phase])
        for k, v in default_noise_params.items()
    )
    params_str = ", ".join(
        ["{} {}".format(noise_name, value) for noise_name, value in params_gen]
    )
    return info_str + params_str


def override_noise_params(noise_type, phase):
    """Provide appropriate default noise params when not passed from cli."""
    try:
        current_noise_params = default_noise_params[noise_type]
    except KeyError as e:
        raise ValueError("Noise type unknown") from e
    param = current_noise_params[phase]

    fmt_str = "{} param not passed. Overriding with default {} for noise type {}"
    print(fmt_str.format(phase.capitalize(), param, noise_type))

    return param


def validate_config(config):
    """Perform additional checks on the configuration."""
    if config.command == "train":

        if config.train_params is None:
            config.train_params = override_noise_params(config.noise_type, "train")

        if config.val_param is None:
            config.val_param = override_noise_params(config.noise_type, "val")

        if config.dataset_root is None:
            config.dataset_root = read_directory_from_env(ENV_VAR_DATASET_ROOT)
            print("Dataset root picked from environment variable")

        if config.use_external_validation:
            if config.val_dataset_root is None:
                config.val_dataset_root = read_directory_from_env(
                    ENV_VAR_VALIDATION_DATASET_ROOT
                )
            print("Validation dataset root picked from environment variable")

    elif config.command == "test":

        if not os.path.isfile(config.test_checkpoint):
            raise ValueError("The checkpoint path is invalid.")

        if config.test_param is None:
            config.test_param = override_noise_params(config.noise_type, "test")

        if config.test_dataset_root is None:
            config.test_dataset_root = read_directory_from_env(
                ENV_VAR_TEST_DATASET_ROOT
            )
            print("Test dataset root picked from environment variable")

    return config


def check_directory_structure(config):
    """Ensure that the right directory structure is used."""
    if config.root_dir != directory_structure.ROOT_DIR:
        directory_structure.update(config.root_dir)

    directory_structure.create()

    if config.command == "test":

        checkpoint_name = os.path.basename(config.test_checkpoint).replace(".pt", "")

        # All the tests for this train run will reside in this directory
        test_base_dir = os.path.join(
            directory_structure.TEST_RUNS_PATH, "test_" + checkpoint_name
        )
        if not os.path.isdir(test_base_dir):
            os.mkdir(test_base_dir)

        # This is the directory for the current run of tests
        current_test_run_path = os.path.join(test_base_dir, TIMESTAMP)
        os.mkdir(current_test_run_path)
        # This is used by tensorboard for its artifacts
        test_log_dir = os.path.join(current_test_run_path, "events")
        os.mkdir(test_log_dir)
        config.test_log_dir = test_log_dir

        # Get the directory of the experiment for the given checkpoint
        experiment_dir = os.path.dirname(os.path.dirname(config.test_checkpoint))
        # This is the configuration used to train the checkpoint to test
        train_config_path = os.path.join(experiment_dir, "run_config.json")
        train_config = read_config_file(train_config_path)

        # Make a copy of the train configuration in the current test directory
        shutil.copy(
            train_config_path, os.path.join(current_test_run_path, "train_config.json")
        )

        test_config = vars(config)

        for key in ["brown_gaussian_std", "noise_type"]:
            if train_config[key] != test_config[key]:
                warnings.warn(
                    "Overriding {}='{}' with the value '{}', "
                    "used for training the checkpoint.".format(
                        key, test_config[key], train_config[key]
                    )
                )
                setattr(config, key, train_config[key])

    return config


def get_config(main_parser, args=None):
    """Parse the configuration for the execution and validate it.

    Parameters
    ----------
    main_parser: arparse.ArgumentParser
        The configured parser instance used for parsing the arguments
    args : list of strings
        Source of configuration. If None, `sys.argv` arguments are analyzed

    Returns
    -------
    argparse.Namespace
        The parsed arguments in a Namespace object

    """
    parsed_args = main_parser.parse_args(args)

    parsed_args = validate_config(parsed_args)

    return parsed_args


def save_config_file(config, path=None):
    """Save the configuration to the experiment directory."""
    if path is None:
        path = directory_structure.RUN_CONFIG_PATH
    with open(path, "w") as f:
        json.dump(vars(config), f, sort_keys=True, indent=4)


def read_config_file(config_path=None):
    """Read the specified configuration file.

    Defaults to the current experiment file.
    """
    if config_path is None:
        config_path = directory_structure.RUN_CONFIG_PATH
    with open(config_path, "r") as f:
        config = json.load(f)

    return config


if __name__ == "__main__":

    configs = get_config(define_parser())
    print(tabulate_config(configs))
