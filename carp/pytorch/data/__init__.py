from __future__ import annotations

import sys
from typing import Any, Dict, Tuple, Iterable, Callable

from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import BatchEncoding
from carp.util import TokMaskTuplePass, TokMaskTupleRev
from typeguard import typechecked


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
class BaseDataPipeline(Dataset):
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

    @staticmethod
    def tokenizer_factory(tokenizer: Callable, process : Callable, context_len: int) -> Callable:
        """Function factory that creates a collate function for use with a torch.util.data.Dataloader

        Args:
            tokenizer (Callable): A Huggingface model tokenizer, taking strings to torch Tensors
            context_len (int): Max length of the passages passed to the tokenizer

        Returns:
            Callable: A function that will take a batch of string tuples and tokenize them properly.
        """

        @typechecked
        def _tok(string_batch: Iterable[str]) -> BatchEncoding:
            for i, _ in enumerate(string_batch):
                if len(string_batch[i]) > context_len:
                    string_batch[i] = string_batch[i][-context_len:]
            if not isinstance(string_batch, list):
                string_batch = list(string_batch)
            return tokenizer(string_batch)

        @typechecked
        def collate(
            data: Iterable[Tuple[str, str]]
        ) -> Tuple[TokMaskTuplePass, TokMaskTupleRev]:
            passages, reviews = zip(*data)
            pass_tokens, rev_tokens = _tok(list(passages)), _tok(list(reviews))
            pass_masks = pass_tokens["attention_mask"]
            rev_masks = rev_tokens["attention_mask"]
            pass_tokens = pass_tokens["input_ids"]
            rev_tokens = rev_tokens["input_ids"]
            return (
                (pass_tokens, pass_masks),
                (rev_tokens, rev_masks),
            )

        return collate
    
from carp.pytorch.data.mlm_pipeline import MLMDataPipeline

def get_datapipeline(name):
    return _DATAPIPELINE[name.lower()]

def get_datapipeline_names():
    return _DATAPIPELINE.keys()
