"""Functions that define train/test procedures."""


def train(dataloader, network, criterion, optimizer, config):
    """Execute the trainig procedure with the passed data and configuration.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The dataloader which provides examples and targets
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
        print("epoch:", epoch)

        network.train()

        train_loss = 0.0
        for batch_index, data in enumerate(dataloader):
            sample = data.sample.to(config.device)
            target = data.target.to(config.device)

            batch_size = len(sample)

            optimizer.zero_grad()

            output = network(sample)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print("Train loss: {:.5f}".format(train_loss / len(dataloader)))


def test(configuration):
    print("test")
