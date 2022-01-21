from torch.functional import Tensor
from carp.pytorch.data import *
from carp.pytorch.model.encoders import BaseEncoder

from dataclasses import dataclass
import torch
from torchtyping import TensorType
from collections import OrderedDict
import ast
from typing import List

import pandas as pd
import numpy as np
from numpy.random import choice

def construct_count_label(label_names_tok : List[str]):
    def count_label(str_rep : str):
        label_counts = OrderedDict([(key,0) for key in label_names_tok])
        l = ast.literal_eval(str_rep)
        for annotator in l:
            if len(annotator) < 1:
                continue
            for annotation in annotator:
                ann = annotation[0].replace('_', ' ')
                if ann in label_counts:
                    label_counts[ann] += 1
        return label_counts
    return count_label

def construct_parse_label(label_names_tok : List[str]):
    def parse_label(str_rep : str):
        #print(type(str_rep))
        labels = []
        l = ast.literal_eval(str_rep)
        for annotator in l:
            if len(annotator) < 1:
                continue
            for annotation in annotator:
                ann = annotation[0].replace('_', ' ')
                if ann in label_names_tok:
                    labels.append(ann)
        return max(set(labels), key=labels.count)
    return parse_label

# number of passages x size of distribution of labels 
@dataclass
class ScarecrowTargetElement:
    target_dist : TensorType["pass_N", -1]

@register_datapipeline
class ScarecrowDataPipeline(BaseDataPipeline):
    
    """Dataset wrapper class to ease working with the CARP dataset and Pytorch data utilities."""
    def __init__(
        self,
        dupe_protection: bool = True,
        path: str = "dataset",
    ):
        # We'll load scarecrow by default but in the future I hope to have a very standardized format
        scarecrow_pd = pd.read_csv('grouped_data.csv')

        keep_labels = ['Off-prompt', 'Grammar_Usage', 'Needs_Google', 'Incoherent', 
                    'Technical_Jargon', 'Redundant']
        label_names = keep_labels

        label_names_tok = [l.replace('_', ' ') for l in label_names]

        scarecrow_pd['counts'] = scarecrow_pd['responses'].apply(construct_count_label(label_names_tok))
        scarecrow_pd['counts_vectors'] = scarecrow_pd['counts'].apply(lambda x: np.fromiter(x.values(),dtype=int))
        scarecrow_pd['counts_distributions'] = scarecrow_pd['counts_vectors'].apply(lambda x: (x+1)/(x.sum()+x.shape[0])) # smoothen counts
        scarecrow_pd['sampled_responses'] = scarecrow_pd['counts_distributions'].apply(lambda x: label_names_tok[choice(np.arange(x.shape[0]), 1, p=x).item()])
        scarecrow_pd['responses'] = scarecrow_pd['responses'].apply(construct_parse_label(label_names_tok))

        while len(scarecrow_pd['responses'][:1200].value_counts()) < len(label_names) or len(scarecrow_pd['responses'][1200:].value_counts()) < len(label_names):
            scarecrow_pd = scarecrow_pd.sample(frac=1)

        # get the passages we want to tune on
        self.passages = list(scarecrow_pd['generation'])
        # get the target distributions
        self.reviews = list(scarecrow_pd['counts_distributions'])
       
        # we do not initialize the super since scarecrow has a very different dataset format than the original data
        #super().__init__(dupe_protection, path)

    @staticmethod
    def tokenizer_factory(_tok : Callable, encoder: BaseEncoder)  -> Callable:
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
            reviews = torch.cat(list(map(\
                lambda x: torch.tensor(x).unsqueeze(0),\
                    list(reviews))), dim=0)

                
            return (
                BatchElement(pass_tokens, pass_masks),
                ScarecrowTargetElement(reviews)
            )

        return collate