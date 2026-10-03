"""Shared instrumented ClientApp for Plain FedAvg and SecAgg+."""

import os
import random
from time import perf_counter

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context, Error, Message

from flwr.common.secure_aggregation.secaggplus_constants import (
    RECORD_KEY_CONFIGS,
    Key,
    Stage,
)

from thesis_compare.measurement import (
    append_message_trace,
    message_nbytes,
)
from thesis_compare.task import (
    get_weights,
    load_data,
    make_net,
    set_weights,
    test,
    train,
)


def _dropout_set(context, round_number):
    """Return deterministic logical clients that drop in this round."""

    dropout_rate = float(
        context.run_config.get("dropout-rate", 0.0)
    )

    num_clients = int(
        context.run_config["num-clients"]
    )

    if dropout_rate <= 0.0:
        return set()

    dropout_count = int(
        round(dropout_rate * num_clients)
    )

    dropout_count = max(
        0,
        min(dropout_count, num_clients - 1),
    )

    seed = int(
        context.run_config["seed"]
    )

    rng = random.Random(
        seed * 1_000_003
        + round_number * 97_409
    )

    return set(
        rng.sample(
            range(num_clients),
            dropout_count,
        )
    )


def protocol_mod(msg, context, call_next):
    """Enable SecAgg+ and deterministic controlled dropout."""

    protocol = str(
        context.run_config["protocol"]
    )

    partition_id = int(
        context.node_config["partition-id"]
    )

    try:
        round_number = int(
            str(msg.metadata.group_id)
        )
    except (TypeError, ValueError):
        round_number = 0

    message_type = str(
        msg.metadata.message_type
    )

    incoming_bytes = message_nbytes(msg)

    secagg_stage = None

    if protocol == "secagg":
        try:
            secagg_stage = (
                msg.content.config_records[
                    RECORD_KEY_CONFIGS
                ][Key.STAGE]
            )
        except (KeyError, AttributeError):
            secagg_stage = None

    # Plain: drop during the normal training request.
    plain_fit = (
        protocol == "plain"
        and message_type.startswith("train")
    )

    # SecAgg+: drop after setup/share-keys, when the
    # masked model vector should be returned.
    secagg_fit = (
        protocol == "secagg"
        and secagg_stage
        == Stage.COLLECT_MASKED_VECTORS
    )

    dropped = (
        (plain_fit or secagg_fit)
        and partition_id
        in _dropout_set(
            context,
            round_number,
        )
    )

    if dropped:
        reason = (
            "Controlled thesis dropout: "
            f"protocol={protocol}, "
            f"round={round_number}, "
            f"partition={partition_id}"
        )

        print(
            f"[CONTROLLED_DROPOUT] {reason}",
            flush=True,
        )

        out_msg = Message(
            Error(
                code=0,
                reason=reason,
            ),
            reply_to=msg,
        )
    elif protocol == "secagg":
        out_msg = secaggplus_mod(
            msg,
            context,
            call_next,
        )
    else:
        out_msg = call_next(
            msg,
            context,
        )

    outgoing_bytes = message_nbytes(
        out_msg
    )

    append_message_trace(
        os.environ.get(
            "THESIS_COMM_LOG",
            "",
        ),
        protocol=protocol,
        partition_id=partition_id,
        message_type=message_type,
        group_id=str(
            msg.metadata.group_id
        ),
        incoming_bytes=incoming_bytes,
        outgoing_bytes=outgoing_bytes,
    )

    return out_msg


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
        """Perform local training and report raw model payload."""

        download_model_bytes = sum(
            np.asarray(array).nbytes
            for array in parameters
        )

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

        updated_parameters = get_weights(self.net)

        upload_model_bytes = sum(
            np.asarray(array).nbytes
            for array in updated_parameters
        )

        return (
            updated_parameters,
            len(self.trainloader.dataset),
            {
                "train_loss": train_loss,
                "local_train_seconds": local_train_seconds,
                "raw_model_download_bytes": download_model_bytes,
                "raw_model_upload_bytes": upload_model_bytes,
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
    """Create deterministic logical client."""

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
