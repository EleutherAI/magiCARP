import argparse
import deepspeed
import math
import torch
import torch.functional as F
from torchtyping import TensorType
from typing import List, Iterable, Any, Tuple

# Break list or tensor into chunks
def chunk(L: TensorType['batch'], sep: int) -> List[TensorType['minibatch']]:
    size = len(L)
    return [
        L[i * sep:min(size, (i+1) * sep)] for i in range(math.ceil(size / sep))
    ]
    

# Generate indices in dataset for batch
def generate_indices(total_size: int, batch_size: int, shuffle: bool = True) -> List[TensorType['minibatch']]:
    inds = torch.randperm(total_size) if shuffle else torch.arange(total_size)
    return chunk(inds, batch_size)


# Scheduling function w/ rampup and decay
def get_scheduling_func(config):
    def lerp(a, b, t):
        t = min(1, t)
        t = max(0, t)
        return a + (b - a) * t
    ratio = config.learning_rate_target / config.learning_rate_init
    def schedule(step):
        if step < config.lr_ramp_steps:
            next_step = (step + 1) / config.lr_ramp_steps
        else:
            next_step = lerp(1, ratio, (step - config.lr_ramp_steps) / config.lr_decay_steps)
        return next_step
    return schedule
        

def list_has_dupes(texts: List[str]) -> bool:
    unique_elems = set(texts)
    return len(unique_elems) != len(texts)

# Check if batch has any duplicate passages or reviews
def batch_has_dupes(pass_batch, rev_batch):
    return list_has_dupes(pass_batch) or list_has_dupes(rev_batch)

def get_arguments():
    parser = argparse.ArgumentParser(description = "CARP")
    parser.add_argument('--backend', type=str, default = 'nccl')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


mbTokens = TensorType["microbatch", -1]


def batch_data(
    data: Iterable[Any], batch_size: int, discard_partial=False
) -> Iterable[Iterable[Any]]:
    """Takes an input of data and returns a batch of data"""
    batch = []
    for x in data:
        batch.append(x)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if not discard_partial and len(batch) > 0:
        yield batch

TokMaskTuplePass = Tuple[TensorType["batch", "pass_N"], TensorType["batch", "pass_N"]]
TokMaskTupleRev = Tuple[TensorType["batch", "rev_N"], TensorType["batch", "rev_N"]]
