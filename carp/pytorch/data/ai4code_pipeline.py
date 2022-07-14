from dataclasses import dataclass

from torchtyping import TensorType
from transformers.data.data_collator import DataCollatorForLanguageModeling

from carp.pytorch.data import *
from carp.pytorch.model.encoders import BaseEncoder
import pandas as pd

@register_datapipeline
class AI4CodeDataPipeline(BaseDataPipeline):

    """Dataset wrapper class to ease working with the CARP dataset and Pytorch data utilities."""

    def __init__(
        self,
        dupe_protection: bool = True,
        path: str = "dataset",
    ):
        dataframe = pd.read_csv("ai4code/clip_train_valid.csv")

        # get the first column
        self.passages = dataframe.iloc[:, 0]
        # get the second column
        self.reviews = dataframe.iloc[:, 1]

        # do not initialize the base case
        #super().__init__(dupe_protection, path)