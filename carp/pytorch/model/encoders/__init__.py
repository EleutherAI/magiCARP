from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Iterable, Union, List
from torchtyping import TensorType

import torch
from torch import nn

from transformers import AutoModel, AutoTokenizer, AutoConfig

# specifies a dictionary of architectures
_ENCODERS: Dict[str, any] = {}  # registry

def register_encoder(name):
    """Decorator used register a CARP encoders 

        Args:
            name: Name of the encoder
    """

    def register_class(cls, name):
        _ENCODERS[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)
    
    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls

def extract_neo(output: Dict[str, Any]) -> TensorType['batch', -1, 'embed_dim']:
    return output["hidden_states"][-2]


def extract_roberta(output: Tuple) -> TensorType['batch', -1, 'embed_dim']:
    return output[0]


Device = Union[str, torch.DeviceObjType]


@dataclass
class BaseEncoderOutput():
    hidden : TensorType['batch', 'embed_dim']


class BaseEncoder(nn.Module):

    # For different models, hidden state is returned differently
    extract_fns = {"neo": extract_neo, "roberta": extract_roberta}

    def __init__(self, model_path: str, model_arch: str, skip_init : bool = False):
        super().__init__()
        self.extract_fn = self.extract_fns.get(model_arch)
        self.cfg = AutoConfig.from_pretrained(model_path)
        self.d_model = self.cfg.hidden_size

        if not skip_init:
            self.model = AutoModel.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            # add quote token to model and tokenizer
            self.tokenizer.add_tokens(["[quote]"])
            self.model.resize_token_embeddings(len(self.tokenizer))

    @property
    def device(self):
        return self.model.device

    @abstractmethod
    def preprocess(self, string_batch: Iterable[str]) -> Iterable[str]:
        pass

    def call_tokenizer(self, string_batch: Iterable[str]):
        return self.tokenizer(
            self.preprocess(string_batch),
            return_tensors='pt',  # Will they ever _not_ be pytorch tensors?
            padding=True,
        )

    def forward(self, inputs_embeds=False, **kwargs):
        assert 'x' in kwargs.keys(), "Error: Input tensor must be specified."

        if inputs_embeds:
            kwargs['inputs_embeds'] = kwargs['x']
        else:
            kwargs['input_ids'] = kwargs['x']
        del kwargs['x']

        return self.model(
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
    # Given masks returns indices of last tokens
    def last_ones(self, t):
        # Multipliying arange by max
        # makes last non zero column have largest number in arange
        t = t * torch.arange(t.shape[1])
        # Then argmax gives index of last non zero column
        t = t.argmax(1)
        return t

from carp.pytorch.model.encoders.pool_encoder import *
from carp.pytorch.model.encoders.mlm_encoder import *

def get_encoder(name):
    return _ENCODERS[name.lower()]

def get_encoder_names():
    return _ENCODERS.keys()
