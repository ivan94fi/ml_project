# pylint: disable=R0912,R0913,R0914,R0915
"""Functions that define train/test procedures."""
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from ml_project.config_parser import TIMESTAMP, directory_structure
from ml_project.gpu_utils import GpuStats
from ml_project.loggers import ConditionalLogger, CounterSubject, IterableSubject
from ml_project.utils import (
    MetricTracker,
    ProgressPrinter,
    create_figure,
    pad,
    psnr_from_mse,
)


def train(  # noqa: C901
    dataloaders, network, criterion, optimizer, lr_scheduler, config
):
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
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler
    config : argparse.Namespace-like
        All the parsed configuration. The exact class does not matter, but the
        options should be available as attributes

    """
    if config.dry_run:
        print("Dry run. Not executing training")
        return

    writer = SummaryWriter(directory_structure.CURRENT_TB_RUN)
    progress_printer = ProgressPrinter(
        config, progress_template="Loss: {:.3f} - PSNR: {:.3f}"
    )
    gpu_handle = GpuStats()

    if config.checkpoint_interval is not None:
        fname_template = os.path.join(
            directory_structure.CHECKPOINTS_PATH, "n2n_" + TIMESTAMP + "_e{}.pt"
        )

    epochs = IterableSubject()
    phases = IterableSubject()
    checkpoint_logger = ConditionalLogger({"step": epochs}, config.checkpoint_interval)
    iterations = CounterSubject()
    image_logger = ConditionalLogger({"step": iterations}, config.log_images)
    additional_logger = ConditionalLogger(
        {"step": iterations, "phase": phases}, config.log_other_metrics, "train"
    )

    for epoch in epochs.iter(range(config.starting_epoch, config.epochs + 1)):
        print("Epoch: {}/{}".format(epoch, config.epochs))

        for phase in phases.iter(["train", "val"]):
            print("- " + phase.capitalize() + " step...")

            if phase == "train":
                network.train()
            else:
                network.eval()

            running_loss = MetricTracker()
            running_psnr = MetricTracker()

            start_time = time.time()
            epoch_start_time = start_time

            progress_printer.reset(phase)

            for batch_index, data in enumerate(iterations.iter(dataloaders[phase])):
                if phase == "val":
                    original_width = data.sample.shape[2]
                    original_height = data.sample.shape[3]
                    data = pad(data)

                sample = data.sample.to(config.device)
                target = data.target.to(config.device)
                batch_size = sample.shape[0]

                progress_printer.update_batch_info(batch_size, batch_index)

                prepare_time = time.time() - start_time

                with torch.set_grad_enabled(phase == "train"):
                    optimizer.zero_grad()
                    output = network(sample)

                    if phase == "val":
                        output = output[:, :, :original_width, :original_height]
                        target = target[:, :, :original_width, :original_height]

                    loss = criterion(output, target)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                process_time = time.time() - start_time - prepare_time

                running_loss.update(loss.item(), batch_size)
                iter_time = prepare_time + process_time
                efficiency = process_time / iter_time
                running_psnr.update(psnr_from_mse(running_loss.last_value), batch_size)

                # Iteration logging
                global_step = epoch * config.batch_numbers[phase] + batch_index
                if image_logger.should_log():
                    tensors = [data.sample, data.target, output]
                    fig = create_figure(tensors, title="epoch:" + str(epoch))
                    writer.add_figure("input-target-output/" + phase, fig, global_step)
                if additional_logger.should_log():
                    writer.add_scalar("Utils/efficiency", efficiency, global_step)
                    writer.add_scalar("Utils/iter_time", iter_time, global_step)
                    if not gpu_handle.no_gpu:
                        used_mem, rate, temp = gpu_handle.get_gpu_stats()
                        writer.add_scalar("Utils/GPU/mem_used", used_mem, global_step)
                        writer.add_scalar("Utils/GPU/util", rate, global_step)
                        writer.add_scalar("Utils/GPU/temp", temp, global_step)

                progress_printer.show_epoch_progress(
                    running_loss.last_value, running_psnr.last_value
                )
                progress_printer.update_bar(batch_size)
                start_time = time.time()

            progress_printer.close_bar()

            # Epoch logging
            epoch_loss = running_loss.average
            epoch_psnr = running_psnr.average
            print(
                "{} concluded. Loss: {:.3f} PSNR: {:.3f}".format(
                    phase.capitalize(), epoch_loss, epoch_psnr
                )
            )
            writer.add_scalar("Metrics/Loss/" + phase, epoch_loss, epoch)
            writer.add_scalar("Metrics/PSNR/" + phase, epoch_psnr, epoch)

            if phase == "train":
                lr_scheduler.step()
                writer.add_scalar("Utils/lr", lr_scheduler.get_last_lr()[0], epoch)
                epoch_time = int(start_time - epoch_start_time)
                writer.add_scalar("Utils/epoch_time", epoch_time, epoch)

            if phase == "val":
                print()

        if checkpoint_logger.should_log():
            checkpoint = {
                "epoch": epoch,
                "net": network.state_dict(),
                "opt": optimizer.state_dict(),
                "sched": lr_scheduler.state_dict(),
            }
            torch.save(checkpoint, fname_template.format(epoch))

    writer.close()
    gpu_handle.close()


def test(dataloader, network, criterion, config):
    """Execute the testing procedure with the passed data and configuration.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The dataloader which provides examples and targets for testing
    network : torch.nn.Module
        The network to train
    criterion : torch.nn.Module
        Loss function used for training
    config : argparse.Namespace-like
        All the parsed configuration. The exact class does not matter, but the
        options should be available as attributes

    """
    if config.dry_run:
        print("Dry run. Not executing testing")
        return

    batch_size = 1
    phase = "test"
    epoch = config.starting_epoch

    writer = SummaryWriter(log_dir=config.test_log_dir)
    progress_printer = ProgressPrinter(
        config, progress_template="Loss: {:.3f} - PSNR: {:.3f}"
    )
    progress_printer.reset(phase)

    iterations = CounterSubject()
    image_logger = ConditionalLogger({"step": iterations}, config.log_images)

    network.eval()

    running_loss = MetricTracker()
    running_psnr = MetricTracker()

    for batch_index, data in enumerate(iterations.iter(dataloader)):
        original_width = data.sample.shape[2]
        original_height = data.sample.shape[3]
        data = pad(data)

        sample = data.sample.to(config.device)
        target = data.target.to(config.device)

        progress_printer.update_batch_info(batch_size, batch_index)

        with torch.set_grad_enabled(False):
            output = network(sample)

            output = output[:, :, :original_width, :original_height]
            target = target[:, :, :original_width, :original_height]

            loss = criterion(output, target)

        running_loss.update(loss.item(), batch_size)
        running_psnr.update(psnr_from_mse(running_loss.last_value), batch_size)

        # Iteration logging
        global_step = batch_index
        if image_logger.should_log():
            tensors = [data.sample, data.target, output]
            fig = create_figure(tensors, title="epoch:" + str(epoch))
            writer.add_figure("input-target-output/" + phase, fig, global_step)

        progress_printer.show_epoch_progress(
            running_loss.last_value, running_psnr.last_value
        )
        progress_printer.update_bar(batch_size)

    progress_printer.close_bar()

    # Epoch logging
    epoch_loss = running_loss.average
    epoch_psnr = running_psnr.average
    print(
        "{} concluded. Loss: {:.3f} PSNR: {:.3f}".format(
            phase.capitalize(), epoch_loss, epoch_psnr
        )
    )
    writer.add_scalar("Metrics/Loss/" + phase, epoch_loss)
    writer.add_scalar("Metrics/PSNR/" + phase, epoch_psnr)

    print()

    writer.close()
