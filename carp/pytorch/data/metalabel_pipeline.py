import ast
from collections import OrderedDict
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
from numpy.random import choice
from torchtyping import TensorType

from carp.pytorch.data import *
from carp.pytorch.model.encoders import BaseEncoder
from carp.pytorch.data.scarecrow_pipeline import ScarecrowTargetElement


def construct_count_label(label_names_tok: List[str]):
    def count_label(str_rep: str):
        label_counts = OrderedDict([(key, 0) for key in label_names_tok])
        l = ast.literal_eval(str_rep)
        for annotator in l:
            if len(annotator) < 1:
                continue
            for annotation in annotator:
                ann = annotation[0].replace("_", " ")
                if ann in label_counts:
                    label_counts[ann] += 1
        return label_counts

    return count_label


def construct_parse_label(label_names_tok: List[str]):
    def parse_label(str_rep: str):
        # print(type(str_rep))
        labels = []
        l = ast.literal_eval(str_rep)
        for annotator in l:
            if len(annotator) < 1:
                continue
            for annotation in annotator:
                ann = annotation[0].replace("_", " ")
                if ann in label_names_tok:
                    labels.append(ann)
        return max(set(labels), key=labels.count)

    return parse_label

@register_datapipeline
class MetalabelDataPipeline(BaseDataPipeline):

    """Dataset wrapper class to ease working with the CARP dataset and Pytorch data utilities."""

    def __init__(
        self,
        dupe_protection: bool = True,
        path: str = "dataset",
    ):
        # We'll load scarecrow by default but in the future I hope to have a very standardized format
        metalabels_pd = pd.read_csv(path)

        # get the passages we want to tune on
        self.passages = list(metalabels_pd["passages"])
        # get the target distributions
        self.reviews = metalabels_pd[[str(i) for i in range(92)]].values.tolist()

    @staticmethod
    def tokenizer_factory(_tok: Callable, encoder: BaseEncoder) -> Callable:
        """Function factory that creates a collate function for use with a torch.util.data.Dataloader

        Args:
            tokenizer (PreTrainedTokenizer): A Huggingface model tokenizer, taking strings to torch Tensors
            context_len (int): Max length of the passages passed to the tokenizer

        Returns:
            Callable: A function that will take a batch of string tuples and tokenize them properly.
        """

        @typechecked
        def collate(
            data: Iterable[Tuple[str, str]]
        ) -> Tuple[BatchElement, ScarecrowTargetElement]:
            passages, reviews = zip(*data)
            pass_tokens = _tok(list(passages))
            pass_masks = pass_tokens["attention_mask"]
            pass_tokens = pass_tokens["input_ids"]

            # concat all the target distributions
            reviews = torch.cat(
                list(map(lambda x: torch.tensor(x).unsqueeze(0), list(reviews))), dim=0
            )

            return (
                BatchElement(pass_tokens, pass_masks),
                ScarecrowTargetElement(reviews),
            )

        return collate
