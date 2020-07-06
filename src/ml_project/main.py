"""Main script of the project; all computations start from here."""

from ml_project.config_parser import parse_config
from ml_project.procedures import test, train

config = parse_config()

print(config)

if config.command == "train":
    train(config)
elif config.command == "test":
    test(config)
