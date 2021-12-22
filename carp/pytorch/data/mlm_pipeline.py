from datasets import load_from_disk
from carp.pytorch.data import BaseDataPipeline, register_datapipeline

from typing import Tuple
from transformers.data.data_collator import DataCollatorForLanguageModeling


@register_datapipeline
class MLMDataPipeline(BaseDataPipeline):
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