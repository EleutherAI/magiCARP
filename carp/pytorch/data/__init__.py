from __future__ import annotations

import sys
from typing import Any, Dict, Tuple, Iterable, Callable

from datasets import load_from_disk
from torch.utils.data import Dataset
from carp.pytorch.data.utils.data_util import create_tok, BatchElement
from typeguard import typechecked
from functools import partial

from carp.pytorch.model.encoders import BaseEncoder

# specifies a dictionary of architectures
_DATAPIPELINE: Dict[str, any] = {}  # registry

def register_datapipeline(name):
    """Decorator used to register a CARP architecture 

    :param name: Name of architecture being registered.
    :type name: str

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
    """Wrapper for Dataset class that eases working with the CARP dataset and Pytorch data utilities.
    
    :param dupe_protection: Filters out any passages or reviews of length less than 8. Use if dataset contains repeated phrases (i.e. "lol") to prevent duplicate encodings
    :type dupe_protection: bool, defaults to True

    :param path: Path to dataset on disk
    :type path: str, defaults to "dataset"

    """
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
    def create_tokenizer_factory(call_tokenizer : Callable, tokenizer_factory : Callable, context_len : int) -> Callable:
        
        """Function creates a callable tokenizer subroutine and uses it to curry the tokenizer factory

        :param call_tokenizer: A function defined within BaseEncoder that outlines a custom encoder processing step.
        :type call_tokenizer: Callable

        :param tokenizer_factory: The factory we wish to initialize.
        :type tokenizer_factory: Callable

        :param context_len: Max context length of any batch element.
        :type context_len: int
        
        :return: A function that creates a factory, which itself take a batch of string tuples then tokenizes them properly.
        :rtype: Callable

        """
        tok_func = create_tok(call_tokenizer, context_len=context_len)
        return partial(tokenizer_factory, tok_func)

    @staticmethod
    def tokenizer_factory(_tok : Callable, encoder: BaseEncoder)  -> Callable:

        """Function factory that creates a collate function for use with a torch.util.data.Dataloader

        :param _tok: A HuggingFace model tokenizer that turns strings to torch tensors.
        :type _tok: Callable

        :param encoder: A CARP base encoder module.
        :type encoder: class:`carp.pytorch.model.encoders.BaseEncoder`

        :return: A function that takes a batch a string tuples then tokenizes them properly.
        :rtype: Callable
        """
        @typechecked
        def collate(
            data: Iterable[Tuple[str, str]]
        ) -> Tuple[BatchElement, BatchElement]:
            passages, reviews = zip(*data)
            pass_tokens, rev_tokens = _tok(list(passages)), _tok(list(reviews))
            pass_masks = pass_tokens["attention_mask"]
            rev_masks = rev_tokens["attention_mask"]
            pass_tokens = pass_tokens["input_ids"]
            rev_tokens = rev_tokens["input_ids"]
            return (
                BatchElement(pass_tokens, pass_masks),
                BatchElement(rev_tokens, rev_masks),
            )

        return collate
    
from carp.pytorch.data.mlm_pipeline import MLMDataPipeline
from carp.pytorch.data.scarecrow_pipeline import ScarecrowDataPipeline

def get_datapipeline(name):
    return _DATAPIPELINE[name.lower()]

def get_datapipeline_names():
    return _DATAPIPELINE.keys()
