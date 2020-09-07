"""Functions that define train/test procedures."""
import torch


def train(dataloader, config):
    """Execute the trainig procedure with the passed data and configuration.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The dataloader which provides examples and targets
    config : argparse.Namespace-like
        All the parsed configuration. The exact class does not matter, but the
        options should be available as attributes

    """
    if config.dry_run:
        print("Dry run. Not executing training")
        return

    for epoch in range(config.epochs):
        print("epoch:", epoch)
        for batch_index, training_pair in enumerate(dataloader):
            sample = training_pair.sample
            target = training_pair.target
            print(batch_index, end=" " if len(sample) < config.batch_size else "\n")
            assert isinstance(sample, torch.Tensor), "sample is not torch tensor"
            assert isinstance(target, torch.Tensor), "target is not torch tensor"
            assert sample.dtype == torch.float32, "sample is not float"
            assert target.dtype == torch.float32, "target is not float"
            actual_batch_size = len(sample)
            correct_shape = (actual_batch_size, 3, config.input_size, config.input_size)
            assert sample.shape == correct_shape, "sample has wrong shape"
            assert target.shape == correct_shape, "target has wrong shape"


def test(configuration):
    print("test")
