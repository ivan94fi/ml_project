# pylint: disable=too-many-locals,too-many-statements
"""Functions that define train/test procedures."""
import time

import torch
from prefetch_generator import BackgroundGenerator
from torch.utils.tensorboard import SummaryWriter

from ml_project.utils import ProgressPrinter, psnr_from_mse


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
    progress_printer = ProgressPrinter(
        config,
        progress_template="Eff: {:.2f} ({:.2f}-{:.2f}) Loss: {:.3f} PSNR: {:.3f}",
    )

    for epoch in range(1, config.epochs + 1):
        print("Epoch: {}/{}".format(epoch, config.epochs))

        for phase in ["train", "val"]:
            print("- " + phase.capitalize() + " step...")

            if phase == "train":
                network.train()
            else:
                network.eval()

            running_loss = 0.0
            running_efficiency = 0.0
            running_psnr = 0.0

            start_time = time.time()
            epoch_start_time = start_time

            dataloader = (
                BackgroundGenerator(dataloaders[phase])
                if config.use_bg_generator
                else dataloaders[phase]
            )

            progress_printer.reset(phase)

            for batch_index, data in enumerate(dataloader):
                sample = data.sample.to(config.device)
                target = data.target.to(config.device)
                batch_size = sample.shape[0]

                progress_printer.update_batch_info(batch_size, batch_index)

                prepare_time = time.time() - start_time

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    output = network(sample)
                    loss = criterion(output, target)
                    psnr = psnr_from_mse(loss.item())

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                process_time = time.time() - start_time - prepare_time

                running_loss += loss.item() * batch_size
                efficiency = process_time / (prepare_time + process_time)
                running_efficiency += efficiency
                running_psnr += psnr

                progress_printer.show_epoch_progress(
                    efficiency, prepare_time, process_time, loss.item(), psnr
                )

                progress_printer.update_bar(batch_size)
                start_time = time.time()

            progress_printer.close_bar()

            # Logging
            epoch_loss = running_loss / config.dataset_sizes[phase]
            epoch_psnr = running_psnr / config.batch_numbers[phase]
            print(
                "{} concluded. Loss: {:.3f} PSNR: {:.3f}".format(
                    phase.capitalize(), epoch_loss, epoch_psnr
                )
            )
            writer.add_scalar("Loss/" + phase, epoch_loss, epoch)
            writer.add_scalar("PSNR/" + phase, epoch_psnr, epoch)

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
