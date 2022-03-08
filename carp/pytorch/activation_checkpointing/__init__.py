from carp.pytorch.activation_checkpointing.checkpoint_engine import (
    ActivationCheckpointingEngine,
)


def checkpoint(num_layers):
    return ActivationCheckpointingEngine(
        cpu_checkpointing=True,
        contiguous_checkpointing=False,
        partitioned_checkpointing=False,
        num_layers=num_layers,
    ).checkpoint
