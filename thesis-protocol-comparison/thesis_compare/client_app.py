"""Shared ClientApp for Plain FedAvg and SecAgg+."""

from time import perf_counter

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context

from thesis_compare.task import (
    get_weights,
    load_data,
    make_net,
    set_weights,
    test,
    train,
)


def protocol_mod(msg, context, call_next):
    """Enable SecAgg+ only when protocol=secagg."""

    protocol = str(
        context.run_config["protocol"]
    )

    if protocol == "secagg":
        return secaggplus_mod(
            msg,
            context,
            call_next,
        )

    return call_next(
        msg,
        context,
    )


class FlowerClient(NumPyClient):
    """Identical ML client for both protocols."""

    def __init__(
        self,
        partition_id,
        trainloader,
        valloader,
        local_epochs,
        learning_rate,
        seed,
    ):
        self.partition_id = partition_id

        self.net = make_net(seed)

        self.trainloader = trainloader
        self.valloader = valloader

        self.local_epochs = local_epochs
        self.learning_rate = learning_rate

        self.device = torch.device(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

    def fit(self, parameters, config):
        """Perform identical local training."""

        set_weights(
            self.net,
            parameters,
        )

        start = perf_counter()

        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.learning_rate,
            self.device,
        )

        local_train_seconds = (
            perf_counter() - start
        )

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {
                "train_loss": train_loss,
                "local_train_seconds": local_train_seconds,
            },
        )

    def evaluate(self, parameters, config):
        """Evaluate the global model."""

        set_weights(
            self.net,
            parameters,
        )

        loss, accuracy = test(
            self.net,
            self.valloader,
            self.device,
        )

        return (
            loss,
            len(self.valloader.dataset),
            {"accuracy": accuracy},
        )


def client_fn(context: Context):
    """Create one deterministic logical client."""

    partition_id = int(
        context.node_config["partition-id"]
    )

    num_partitions = int(
        context.node_config["num-partitions"]
    )

    batch_size = int(
        context.run_config["batch-size"]
    )

    seed = int(
        context.run_config["seed"]
    )

    trainloader, valloader = load_data(
        partition_id,
        num_partitions,
        batch_size,
        seed,
    )

    return FlowerClient(
        partition_id=partition_id,
        trainloader=trainloader,
        valloader=valloader,
        local_epochs=int(
            context.run_config["local-epochs"]
        ),
        learning_rate=float(
            context.run_config["learning-rate"]
        ),
        seed=seed,
    ).to_client()


app = ClientApp(
    client_fn=client_fn,
    mods=[protocol_mod],
)
