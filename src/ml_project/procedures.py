# pylint: disable=too-many-locals,too-many-statements
"""Functions that define train/test procedures."""
import os
import time

import torch
from prefetch_generator import BackgroundGenerator
from torch.utils.tensorboard import SummaryWriter

from ml_project.utils import (
    ProgressPrinter,
    checkpoint_fname_template,
    create_figure,
    get_gpu_stats,
    get_nvml_handle,
    nvml_shutdown,
    psnr_from_mse,
)


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

    writer = SummaryWriter()  # TODO: use a tmp dir?
    progress_printer = ProgressPrinter(
        config, progress_template="Loss: {:.3f} - PSNR: {:.3f}"
    )
    handle = get_nvml_handle()

    if config.checkpoint_interval is not None:
        if not os.path.isdir(config.checkpoints_root):
            raise ValueError("The checkpoint root is not a valid directory")
        save_dir = os.path.join(config.checkpoints_root, "checkpoints")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        fname_template = os.path.join(save_dir, checkpoint_fname_template())

    for epoch in range(config.starting_epoch, config.epochs + 1):
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
                if config.bg_generator
                else dataloaders[phase]
            )

            progress_printer.reset(phase)

            for batch_index, data in enumerate(dataloader):
                sample = data.sample.to(config.device)
                target = data.target.to(config.device)
                batch_size = sample.shape[0]

                progress_printer.update_batch_info(batch_size, batch_index)

                prepare_time = time.time() - start_time

                with torch.set_grad_enabled(phase == "train"):
                    optimizer.zero_grad()
                    output = network(sample)
                    loss = criterion(output, target)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                process_time = time.time() - start_time - prepare_time

                current_loss = loss.item()
                running_loss += current_loss * batch_size
                efficiency = process_time / (prepare_time + process_time)
                running_efficiency += efficiency
                psnr = psnr_from_mse(current_loss)
                running_psnr += psnr * batch_size

                progress_printer.show_epoch_progress(current_loss, psnr)

                # Iteration logging
                step = epoch * config.batch_numbers[phase] + batch_index
                if (
                    config.log_images is not None
                    and batch_index % config.log_images == 0
                ):
                    with torch.set_grad_enabled(False):
                        fig = create_figure(
                            [data.sample, data.target, output],
                            title="epoch:" + str(epoch),
                        )
                    writer.add_figure(
                        "input-target-output/" + phase, fig, global_step=step
                    )
                if phase == "train":
                    if (
                        config.log_other_metrics is not None
                        and batch_index % config.log_other_metrics == 0
                    ):
                        used_mem, rate, temp = get_gpu_stats(handle)
                        additional_metrics = {
                            "Utils/efficiency": efficiency,
                            "Utils/iter_time": prepare_time + process_time,
                            "Utils/GPU/mem_used": used_mem,
                            "Utils/GPU/util": rate,
                            "Utils/GPU/temp": temp,
                        }
                        for tag, value in additional_metrics.items():
                            writer.add_scalar(tag, value, global_step=step)

                progress_printer.update_bar(batch_size)
                start_time = time.time()

            progress_printer.close_bar()

            # Epoch logging
            epoch_loss = running_loss / config.dataset_sizes[phase]
            epoch_psnr = running_psnr / config.dataset_sizes[phase]
            print(
                "{} concluded. Loss: {:.3f} PSNR: {:.3f}".format(
                    phase.capitalize(), epoch_loss, epoch_psnr
                )
            )
            writer.add_scalar("Metrics/Loss/" + phase, epoch_loss, epoch)
            writer.add_scalar("Metrics/PSNR/" + phase, epoch_psnr, epoch)
            if phase == "train":
                epoch_time = int(start_time - epoch_start_time)
                writer.add_scalar("Utils/epoch_time", epoch_time, epoch)

            if phase == "val":
                print()

        if (
            config.checkpoint_interval is not None
            and epoch % config.checkpoint_interval == 0
        ):
            checkpoint = {
                "epoch": epoch,
                "net": network.state_dict(),
                "opt": optimizer.state_dict(),
            }
            torch.save(checkpoint, fname_template.format(epoch))

    writer.close()
    nvml_shutdown()


def test(configuration):
    print("test")
