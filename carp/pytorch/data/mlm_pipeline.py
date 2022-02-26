from dataclasses import dataclass

from torchtyping import TensorType
from transformers.data.data_collator import DataCollatorForLanguageModeling

from carp.pytorch.data import *
from carp.pytorch.model.encoders import BaseEncoder


@dataclass
class MLMBatchElement(BatchElement):
    mlm_input_ids: TensorType[-1, "pass_N"]
    mlm_labels: TensorType[-1, "pass_N"]


@register_datapipeline
class MLMDataPipeline(BaseDataPipeline):

    """Dataset wrapper class to ease working with the CARP dataset and Pytorch data utilities."""

    def __init__(
        self,
        dupe_protection: bool = True,
        path: str = "dataset",
    ):
        super().__init__(dupe_protection, path)

    @staticmethod
    def tokenizer_factory(_tok: Callable, encoder: BaseEncoder) -> Callable:
        """Function factory that creates a collate function for use with a torch.util.data.Dataloader

        Args:
            tokenizer (PreTrainedTokenizer): A Huggingface model tokenizer, taking strings to torch Tensors
            context_len (int): Max length of the passages passed to the tokenizer

        Returns:
            Callable: A function that will take a batch of string tuples and tokenize them properly.
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
