from __future__ import annotations

import sys
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Iterable, Callable, List
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

import torch
import torch.nn.functional as F
from torch import nn
from datasets import load_from_disk
from torch.utils.data import Dataset


# specifies a dictionary of architectures
_DATAPIPELINE: Dict[str, any] = {}  # registry

def register_datapipeline(name):
    """Decorator used register a CARP architecture 

        Args:
            name: Name of the architecture
    """

    def register_class(cls, name):
        _DATAPIPELINE[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)
    
    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls


@register_datapipeline
class BaseDatapipeline(Dataset):
    """Dataset wrapper class to ease working with the CARP dataset and Pytorch data utilities."""

    def __init__(
        self,
        dupe_protection: bool = True,
        path: str = "dataset",
    ):
        dataset = load_from_disk(path)
        train = dataset["train"]
        passages = train["story_target"]
        reviews = train["target_comment"]
        if dupe_protection:
            size = len(passages)
            i = 0
            while i < size:
                if len(reviews[i]) <= 7 or len(passages[i]) <= 7:
                    del passages[i]
                    del reviews[i]
                    size -= 1
                else:
                    i += 1
        self.passages = passages
        self.reviews = reviews

    def __getitem__(self, index: int) -> Tuple[str, str]:
        return self.passages[index], self.reviews[index]

    def __len__(self) -> int:
        return len(self.passages)
    





from carp.pytorch.model.architectures.carp import CARP
from carp.pytorch.model.architectures.carp_momentum import CARPMomentum
from carp.pytorch.model.architectures.carp_cloob import CARPCloob

def get_datapipeline(name):
    return _DATAPIPELINE[name.lower()]

def get_datapipeline_names():
    return _DATAPIPELINE.keys()
