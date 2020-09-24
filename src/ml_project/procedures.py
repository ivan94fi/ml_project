# pylint: disable=too-many-locals
"""Functions that define train/test procedures."""
import time

import torch
from prefetch_generator import BackgroundGenerator
from torch.utils.tensorboard import SummaryWriter

from ml_project.utils import should_print


def train(dataloaders, network, criterion, optimizer, config):
    """Execute the trainig procedure with the passed data and configuration.

    Parameters
    ----------
    dataloaders : dict of torch.utils.data.DataLoader
        The dataloaders which provides examples and targets for training and validation
    network : torch.nn.Module
        The network to train
    criterion : torch.nn.Module
        Loss function for training
    optimizer : torch.optim.Optimizer
        The optimizer instance used for training
    config : argparse.Namespace-like
        All the parsed configuration. The exact class does not matter, but the
        options should be available as attributes

    """
    if config.dry_run:
        print("Dry run. Not executing training")
        return

    # TODO: pass writer as parameter?
    writer = SummaryWriter()  # TODO: use a tmp dir?

    for epoch in range(config.epochs):
        print("epoch: {}/{}".format(epoch + 1, config.epochs))

        for phase in ["train", "val"]:
            if phase == "train":
                network.train()
            else:
                network.eval()

            running_loss = 0.0
            running_efficiency = 0.0

            start_time = time.time()
            epoch_start_time = start_time

            dataloader = (
                BackgroundGenerator(dataloaders[phase])
                if config.use_bg_generator
                else dataloaders[phase]
            )
            for batch_index, data in enumerate(dataloader):
                sample = data.sample.to(config.device)
                target = data.target.to(config.device)
                batch_size = sample.shape[0]

                prepare_time = time.time() - start_time

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    output = network(sample)
                    loss = criterion(output, target)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * batch_size

                process_time = time.time() - start_time - prepare_time

                efficiency = process_time / (prepare_time + process_time)
                running_efficiency += efficiency

                if should_print(phase, batch_index, config):
                    print(
                        "[{}/{}] Eff: {:.2f} ({:.2f}-{:.2f}) Loss: {:.3f}".format(
                            (batch_index * config.batch_size) + batch_size,
                            config.dataset_sizes[phase],
                            efficiency,
                            prepare_time,
                            process_time,
                            loss.item(),
                        )
                    )
                start_time = time.time()

            # Logging
            epoch_loss = running_loss / config.dataset_sizes[phase]
            print("{} loss: {:.3f}".format(phase.capitalize(), epoch_loss))
            writer.add_scalar("Loss/" + phase, epoch_loss, epoch)

            if phase == "train":
                epoch_time = time.time() - epoch_start_time
                epoch_efficiency = running_efficiency / config.batch_numbers["train"]
                writer.add_scalar("Time", epoch_time, epoch)
                writer.add_scalar("Efficiency", epoch_efficiency, epoch)

            if phase == "val":
                print()

    writer.close()


def test(configuration):
    print("test")
