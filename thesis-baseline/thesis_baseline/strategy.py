"""Deterministic FedAvg strategy for reproducible thesis experiments."""

import random
from collections.abc import Iterable
from time import sleep

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    RecordDict,
)
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg


class DeterministicFedAvg(FedAvg):
    """FedAvg with deterministic logical-client participation."""

    def __init__(self, sampling_seed: int, **kwargs):
        super().__init__(**kwargs)
        self.sampling_seed = sampling_seed

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """Select deterministic logical clients for each training round."""

        if self.fraction_train == 0.0:
            return []

        # Wait until all required simulated nodes are available
        while True:
            all_node_ids = sorted(grid.get_node_ids())

            if len(all_node_ids) >= self.min_available_nodes:
                break

            sleep(1)

        num_nodes = int(len(all_node_ids) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)

        if sample_size > len(all_node_ids):
            raise RuntimeError(
                f"Requested {sample_size} training clients, "
                f"but only {len(all_node_ids)} nodes are available."
            )

        # Deterministic RNG: same seed + same round => same logical clients
        rng = random.Random(self.sampling_seed + server_round)

        logical_client_ids = sorted(
            rng.sample(range(len(all_node_ids)), sample_size)
        )

        # Runtime Flower node IDs may change between simulations.
        # We therefore map stable logical client IDs to current runtime nodes.
        selected_node_ids = [
            all_node_ids[logical_id]
            for logical_id in logical_client_ids
        ]

        print(
            f"[SCHEDULE] round={server_round} "
            f"logical_client_ids={logical_client_ids}",
            flush=True,
        )

        messages = []

        for node_id, logical_client_id in zip(
            selected_node_ids,
            logical_client_ids,
            strict=True,
        ):
            client_config = ConfigRecord(
                {key: value for key, value in config.items()}
            )

            client_config["server-round"] = server_round
            client_config["logical-client-id"] = logical_client_id

            record = RecordDict(
                {
                    self.arrayrecord_key: arrays,
                    self.configrecord_key: client_config,
                }
            )

            message = Message(
                content=record,
                message_type=MessageType.TRAIN,
                dst_node_id=node_id,
            )

            messages.append(message)

        return messages
