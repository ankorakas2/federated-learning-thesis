"""Shared deterministic CIFAR-10 task for protocol comparison."""

import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    """CNN used for both Plain FedAvg and SecAgg+."""

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


def set_seed(seed: int):
    """Set deterministic ML random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True)


def make_net(seed: int = 42):
    """Create deterministically initialized model."""

    set_seed(seed)
    return Net()


def get_weights(net):
    """Return model parameters as NumPy arrays."""

    return [
        value.cpu().numpy()
        for _, value in net.state_dict().items()
    ]


def set_weights(net, parameters):
    """Load NumPy model parameters."""

    params_dict = zip(
        net.state_dict().keys(),
        parameters,
    )

    state_dict = OrderedDict(
        {
            key: torch.tensor(value)
            for key, value in params_dict
        }
    )

    net.load_state_dict(
        state_dict,
        strict=True,
    )


fds = None
fds_seed = None
fds_num_partitions = None


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    seed: int,
):
    """Load deterministic IID CIFAR-10 partition."""

    global fds, fds_seed, fds_num_partitions

    if (
        fds is None
        or fds_seed != seed
        or fds_num_partitions != num_partitions
    ):
        partitioner = IidPartitioner(
            num_partitions=num_partitions
        )

        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
            seed=seed,
        )

        fds_seed = seed
        fds_num_partitions = num_partitions

    partition = fds.load_partition(partition_id)

    client_seed = seed + partition_id

    partition = partition.train_test_split(
        test_size=0.2,
        seed=client_seed,
    )

    transforms = Compose(
        [
            ToTensor(),
            Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
        ]
    )

    def apply_transforms(batch):
        batch["img"] = [
            transforms(image)
            for image in batch["img"]
        ]
        return batch

    partition = partition.with_transform(
        apply_transforms
    )

    generator = torch.Generator()
    generator.manual_seed(client_seed)

    trainloader = DataLoader(
        partition["train"],
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    valloader = DataLoader(
        partition["test"],
        batch_size=batch_size,
    )

    return trainloader, valloader


def train(
    net,
    trainloader,
    epochs,
    learning_rate,
    device,
):
    """Train local model."""

    net.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=learning_rate,
        momentum=0.9,
    )

    net.train()

    running_loss = 0.0
    batches = 0

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = net(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

    return running_loss / batches


def test(net, testloader, device):
    """Evaluate model."""

    net.to(device)
    net.eval()

    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    loss = 0.0

    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            outputs = net(images)

            loss += criterion(
                outputs,
                labels,
            ).item()

            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)

    return loss, accuracy
