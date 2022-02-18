from torch.functional import Tensor
from carp.pytorch.data import *
from carp.pytorch.model.encoders import BaseEncoder
from transformers.data.data_collator import DataCollatorForLanguageModeling

from dataclasses import dataclass
from torchtyping import TensorType


@dataclass
class MLMBatchElement(BatchElement):
    """ 
    Wrapper for BatchElement intended for MLM tasks. Contains same attributes in addition to tokens to input for MLM task and MLM mask labels.
    """
    mlm_input_ids : TensorType[-1, "pass_N"]
    mlm_labels : TensorType[-1, "pass_N"]


@register_datapipeline
class MLMDataPipeline(BaseDataPipeline):
    
    """Wrapper for Dataset class that specifically eases working with CARP MLM
    
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
        super().__init__(dupe_protection, path)

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
        mlm_collator = DataCollatorForLanguageModeling(encoder.tokenizer, mlm=True)

        @typechecked
        def collate(
            data: Iterable[Tuple[str, str]]
        ) -> Tuple[MLMBatchElement, MLMBatchElement]:
            passages, reviews = zip(*data)
            pass_tokens, rev_tokens = _tok(list(passages)), _tok(list(reviews))
            pass_masks = pass_tokens["attention_mask"]
            rev_masks = rev_tokens["attention_mask"]
            pass_tokens = pass_tokens["input_ids"]
            rev_tokens = rev_tokens["input_ids"]

            mlm_pass_tokens, pass_labels = mlm_collator.torch_mask_tokens(pass_tokens)
            mlm_rev_tokens, rev_labels = mlm_collator.torch_mask_tokens(rev_tokens)

            return (
                MLMBatchElement(pass_tokens, pass_masks, mlm_pass_tokens, pass_labels),
                MLMBatchElement(rev_tokens, rev_masks, mlm_rev_tokens, rev_labels),
            )

        return collate