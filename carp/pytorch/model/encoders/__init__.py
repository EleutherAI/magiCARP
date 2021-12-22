from __future__ import annotations

import sys
from dataclasses import dataclass
from abc import ABC, abstractmethod
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


class BaseEncoder(nn.Module):

    # For different models, hidden state is returned differently
    extract_fns = {"neo": extract_neo, "roberta": extract_roberta}

    def __init__(self, model_path: str, model_arch: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.extract_fn = self.extract_fns.get(model_arch)
        self.cfg = AutoConfig.from_pretrained(model_path)
        self.d_model = self.cfg.hidden_size
        
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

    # Given masks returns indices of last tokens
    def last_ones(self, t):
        # Multipliying arange by max
        # makes last non zero column have largest number in arange
        t = t * torch.arange(t.shape[1])
        # Then argmax gives index of last non zero column
        t = t.argmax(1)
        return t

# Abstract base for a model that can serve both for MLM and encoding
# Makes assumption that central model is MLM (i.e. it's roberta)
class MLMEncoder(nn.Module):

    def __init__(self, model_path: str, model_arch: str):
        super().__init__()
        self.model = transformers.RobertaForMaskedLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # default roberta extract doesnt work
        self.extract_fn = lambda out: out['hidden_states'][-1]

        self.cfg = AutoConfig.from_pretrained(model_path)
        self.d_model = self.cfg.hidden_size
        
        # add quote token to model and tokenizer
        self.tokenizer.add_tokens(["[quote]"])
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.encode_flag = False

    def flag_encode(self): self.encode_flag = True
    def flag_mlm(self): self.encode_flag = False

    @property
    def device(self):
        return self.model.device

    @abstractmethod
    def preprocess(self, string_batch: Iterable[str]) -> Iterable[str]:
        pass

    @abstractmethod
    def process_hidden_state(self, hidden: TensorType['batch', 'N', 'embed_dim'], mask: TensorType['batch', 'N']) -> TensorType['batch', 'embed_dim']:
        pass

    def tok(self, string_batch: Iterable[str]):
        return self.tokenizer(
            self.preprocess(string_batch),
            return_tensors='pt',  # Will they ever _not_ be pytorch tensors?
            padding=True,
        )

    def forward(self, x, mask = None):
        out = self.model(
                input_ids = x, 
                attention_mask = mask,
                output_hidden_states = True,
        )
        if self.encode_flag:
            hidden: TensorType['batch', 'N', 'embed_dim'] = self.extract_fn(out)
            return self.process_hidden_state(hidden, mask)
        else:
            logits: TensorType['batch', 'vocab'] = out['logits'] 
            return logits

from carp.pytorch.model.encoders.encoder import *

def get_encoder(name):
    return _ENCODERS[name.lower()]

def get_encoder_names():
    return _ENCODERS.keys()
