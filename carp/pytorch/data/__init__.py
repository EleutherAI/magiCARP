from __future__ import annotations

import sys
from functools import partial
from typing import Callable, Dict, Iterable, Tuple

from datasets import load_from_disk
from torch.utils.data import Dataset
from typeguard import typechecked

from carp.pytorch.data.utils.data_util import BatchElement, create_tok
from carp.pytorch.model.encoders import BaseEncoder

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
        if path is not None:
            dataset = load_from_disk(path)
            train = dataset["train"]
            self.passages = train["story_target"]
            self.reviews = train["target_comment"]
        if dupe_protection:
            size = len(self.passages)
            i = 0
            while i < size:
                if len(self.reviews[i]) <= 7 or len(self.passages[i]) <= 7:
                    del self.passages[i]
                    del self.reviews[i]
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
    def create_tokenizer_factory(
        call_tokenizer: Callable, tokenizer_factory: Callable, context_len: int
    ) -> Callable:

        """Function creates a callable tokenizer subroutine and uses it to curry the tokenizer factory

        Args:
            call_tokenizer (Callable): A function defined within BaseEncoder that outlines a custom encoder processing step
            tokenizer_factory (Callable): The factory we wish to initialize
            context_len (int): Max context length of a batch element.
        Returns:
            Callable: A function that create a factory that will take a batch of string tuples and tokenize them properly.
        """
        tok_func = create_tok(call_tokenizer, context_len=context_len)
        return partial(tokenizer_factory, tok_func)

    @staticmethod
    def tokenizer_factory(_tok: Callable, encoder: BaseEncoder) -> Callable:

        """Function factory that creates a collate function for use with a torch.util.data.Dataloader

        Args:
            _tok (Callable): A Huggingface model tokenizer, taking strings to torch Tensors
            encoder (BaseEncoder): A CARP base encoder module.
        Returns:
            Callable: A function that will take a batch of string tuples and tokenize them properly.
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


from carp.pytorch.data.metalabel_pipeline import MetalabelDataPipeline
from carp.pytorch.data.mlm_pipeline import MLMDataPipeline
from carp.pytorch.data.scarecrow_pipeline import ScarecrowDataPipeline
from carp.pytorch.data.ai4code_pipeline import AI4CodeDataPipeline


def get_datapipeline(name):
    return _DATAPIPELINE[name.lower()]


def get_datapipeline_names():
    return _DATAPIPELINE.keys()
