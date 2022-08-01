from dataclasses import dataclass

import pandas as pd
from torchtyping import TensorType
from transformers.data.data_collator import DataCollatorForLanguageModeling

from carp.pytorch.data import *
from carp.pytorch.model.encoders import BaseEncoder


@dataclass
class AI4CodeHardNegative(BatchElement):
    negative_input_ids: TensorType[-1, "pass_N"]


@register_datapipeline
class AI4CodeDataPipeline(BaseDataPipeline):

    """Dataset wrapper class to ease working with the CARP dataset and Pytorch data utilities."""

    def __init__(
        self,
        dupe_protection: bool = True,
        path: str = "dataset",
    ):
        dataframe = pd.read_csv("../clip_combined.csv")

        # get the first column
        self.passages = list(dataframe.iloc[:, 0])
        # get the second column
        self.reviews = list(dataframe.iloc[:, 1])

        self.passages = list(map(str, self.passages))
        self.reviews = list(map(str, self.reviews))

        # do not initialize the base case
        super().__init__(True, None)
