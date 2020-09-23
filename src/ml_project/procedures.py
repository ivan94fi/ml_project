"""Functions that define train/test procedures."""
import torch


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

    for epoch in range(config.epochs):
        print("epoch: {}/{}".format(epoch + 1, config.epochs))

        for phase in ["train", "val"]:
            if phase == "train":
                network.train()
            else:
                network.eval()

            running_loss = 0.0

            for batch_index, data in enumerate(dataloaders[phase]):
                sample = data.sample.to(config.device)
                target = data.target.to(config.device)
                batch_size = sample.shape[0]

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    output = network(sample)
                    loss = criterion(output, target)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * batch_size

                if phase == "train" and batch_index % 5 == 0:
                    print(
                        "batch {}/{}. Loss: {}".format(
                            batch_index, len(dataloaders[phase]), loss.item()
                        )
                    )

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print("{} loss: {:.5f}".format(phase.capitalize(), epoch_loss))
            if phase == "val":
                print()


def test(configuration):
    print("test")
