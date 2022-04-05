from torch.functional import Tensor
from carp.pytorch.data import *
from carp.pytorch.model.encoders import BaseEncoder
from transformers.data.data_collator import DataCollatorForLanguageModeling
from carp.pytorch.data.utils.data_util import read_dataset_component, read_paraphrase_component

from dataclasses import dataclass
from torchtyping import TensorType
from typing import List
import torch
import os

#TODO:
'''Custom chunk_batch_element
'''


@dataclass
class DistillBatchElement(BatchElement):
    #Reducing over critiques for same stories
    #reduction_matrix : TensorType["pass_N", -1]
    reviews_per_passage: int


@register_datapipeline
class DistillDataPipeline(BaseDataPipeline):

    """Dataset wrapper class to ease working with the CARP dataset and Pytorch data utilities."""
    def __init__(
        self,
        #Prevents duplicates of multiple stories
        dupe_protection: bool = True,
        path: str = "dataset",
    ):
        crit_data = []
        crit_datapath = path+'/paraphrase_train_crits'
        files = os.listdir(crit_datapath)
        files.sort()
        for file in files:
            print(file)
            datapath = os.path.join(crit_datapath,file)
            crit_data_chunk = read_paraphrase_component(datapath)
            crit_data+=crit_data_chunk

        orig_crit_file = 'train_crits.csv'
        orig_crit_datapath = os.path.join(path, orig_crit_file)
        orig_crit_data = read_dataset_component(orig_crit_datapath)
        for orig_crit, crits in zip(orig_crit_data, crit_data):
            crits.append(orig_crit)
        self.reviews_list = crit_data

        story_file = 'train_stories.csv'
        story_datapath = os.path.join(path, story_file)
        story_data = read_dataset_component(story_datapath)
        self.passages = story_data


        # prune to the last 3
        self.reviews_list = list(map(lambda x: [x[-1]], self.reviews_list))

        print("NUM STORIES: ", len(self.passages))
        print("NUM CRITIQUE LISTS: ", len(self.reviews_list))
        print("NUM CRITIQUES PER: ", len(self.reviews_list[0]))


    #Overload for data format (passage, [crit_1,...,crit_n])
    def __getitem__(self, index: int) -> Tuple[str, List[str]]:
            return self.passages[index], self.reviews_list[index]

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
            data: Iterable[Tuple[str, List[str]]]
        ) -> Tuple[BatchElement, DistillBatchElement]:
            #Expects us to double reviews beforehand: passing in list of critiques for each story
            passages, review_lists = zip(*data)
            reviews_per_passage = len(review_lists[0])
            reviews = [review for review_list in review_lists for review in review_list]
            pass_tokens, rev_tokens = _tok(list(passages)), _tok(list(reviews))
            pass_masks = pass_tokens["attention_mask"]
            rev_masks = rev_tokens["attention_mask"]
            pass_tokens = pass_tokens["input_ids"]
            rev_tokens = rev_tokens["input_ids"]

            #eduction_matrix = torch.arange(0, rev_tokens.size()[0], step=1).reshape(-1, reviews_per_passage)

            return (
                BatchElement(pass_tokens, pass_masks),
                DistillBatchElement(rev_tokens, rev_masks, reviews_per_passage),
            )

        return collate
