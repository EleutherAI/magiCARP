from carp.pytorch.data import *
from carp.pytorch.model.encoders import BaseEncoder
from transformers.data.data_collator import DataCollatorForLanguageModeling

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
    def tokenizer_factory(_tok : Callable, encoder: BaseEncoder)  -> Callable:
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
        ) -> Tuple[TokMaskTuplePass, TokMaskTupleRev]:
            passages, reviews = zip(*data)
            pass_tokens, rev_tokens = _tok(list(passages)), _tok(list(reviews))
            pass_masks = pass_tokens["attention_mask"]
            rev_masks = rev_tokens["attention_mask"]
            pass_tokens = pass_tokens["input_ids"]
            rev_tokens = rev_tokens["input_ids"]

            pass_tokens, pass_labels = mlm_collator.torch_mask_tokens(pass_tokens)
            rev_tokens, rev_labels = mlm_collator.torch_mask_tokens(rev_tokens)

            return (
                (pass_tokens, pass_labels, pass_masks),
                (rev_tokens, rev_labels, rev_masks),
            )

        return collate