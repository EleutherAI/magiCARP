from torchtyping import TensorType
from transformers.tokenization_utils_base import BatchEncoding
from typing import Callable, Dict, Iterable, Tuple
from typeguard import typechecked

TokMaskTuplePass = Tuple[TensorType["batch", "pass_N"], TensorType["batch", "pass_N"]]
TokMaskTupleRev = Tuple[TensorType["batch", "rev_N"], TensorType["batch", "rev_N"]]

def tokenizer_factory(tokenizer: Callable, context_len: int) -> Callable:
    """Function factory that creates a collate function for use with a torch.util.data.Dataloader

    Args:
        tokenizer (Callable): A Huggingface model tokenizer, taking strings to torch Tensors
        context_len (int): Max length of the passages passed to the tokenizer

    Returns:
        Callable: A function that will take a batch of string tuples and tokenize them properly.
    """

    @typechecked
    def _tok(string_batch: Iterable[str]) -> BatchEncoding:
        for i, _ in enumerate(string_batch):
            if len(string_batch[i]) > context_len:
                string_batch[i] = string_batch[i][-context_len:]
        if not isinstance(string_batch, list):
            string_batch = list(string_batch)
        return tokenizer(string_batch)

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
        return (
            (pass_tokens, pass_masks),
            (rev_tokens, rev_masks),
        )

    return collate
