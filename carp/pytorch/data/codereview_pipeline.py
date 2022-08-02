import json
import random
import re
import sys
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from torchtyping import TensorType
from transformers.data.data_collator import DataCollatorForLanguageModeling

from carp.pytorch.data import *
from carp.pytorch.model.encoders import BaseEncoder


def split(kv_pair: Tuple[str, str]) -> List[Tuple[str, str]]:
    """
    If the k,v \in kv has a [START] and [END] token, turn it into its own aligned pair
    :param kv_pair: (k, v)
    :return: list of (k, v)
    """
    k, v = kv_pair
    if "[START" in k and "[END" in v:
        tuple_list = []
        # it contains multiple tuples, so extract them.

        # first determine how many tuples there are.
        idx = 0
        while "[START-" + str(idx) + "]" in k:
            idx += 1
        return tuple_list
    return [(k, v)]


@register_datapipeline
class CodeReviewPipeline(BaseDataPipeline):

    """Dataset wrapper class to ease working with the CARP dataset and Pytorch data utilities."""

    def __init__(
        self,
        dupe_protection: bool = True,
        path: str = "dataset",
    ):

        # load json
        # with open("datastack.json") as f:
        #    self.data_stack = json.load(f)

        dataframe = pd.read_csv("extracted_question_critique_pairs_v2.csv")

        dataframe = dataframe[
            dataframe[
                "accepted_answer_paragraph_identifier_annotated_question_text"
            ].notna()
        ]
        dataframe = dataframe[
            dataframe["accepted_answer_paragraph_markdown_masked"].notna()
        ]

        self.data_stack = {}

        for index, row in dataframe.iterrows():
            code = row["accepted_answer_paragraph_identifier_annotated_question_text"]
            answer = row["accepted_answer_paragraph_markdown_masked"]

            if code not in self.data_stack:
                self.data_stack[code] = [answer]
            else:
                self.data_stack[code].append(answer)

        df_temp = self.data_stack.copy()
        for k, v in df_temp.items():
            if not ("[START-" in str(k)):
                continue
            k_new = re.sub(r"\[START-[0-9]+\]", "[START]", k)
            k_new = re.sub(r"\[END-[0-9]+\]", "[END]", k_new)

            del self.data_stack[k]
            self.data_stack[k_new] = v

        # save datastack to json
        with open("datastack.json", "w") as f:
            json.dump(self.data_stack, f)

        self.passages = list(self.data_stack.keys())
        self.reviews = list(self.data_stack.values())

    def __getitem__(self, index: int) -> Tuple[str, str]:
        # we run augmentations every time we fetch a new item.
        # This is only ran 2048 times per step over multiple cpu threads, so it's not a big deal.

        # fetch passage
        passage = self.passages[index]

        # take a random (max) 1200 word span of passage
        passage_split = passage.split()
        if len(passage_split) >= 1200:
            start = min(
                int(len(passage_split) * np.random.rand()), len(passage_split) - 1200
            )
            end = min(int(start + 1200 * np.random.rand()), len(passage_split))
            passage = " ".join(passage_split[start:end])

        # pop from the top and then push to the bottom
        critique = self.reviews[index].pop(0)

        # reinsert critique at a random point in self.reviews[index]
        self.reviews[index].insert(
            random.randint(0, len(self.reviews[index])), critique
        )

        return passage, critique

    def __len__(self) -> int:
        return len(self.data_stack)
