import math
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

import torch
from torchtyping import TensorType
from transformers.tokenization_utils_base import BatchEncoding
from typeguard import typechecked


def check_char(char):
    """Check if char can be encoded"""
    try:
        char.encode("charmap")
        return True
    except:
        return False


def partition_review(rev: Optional[str]) -> List[str]:
    """Takes reviews from raw data format to a list of separate reviews.
    [REDACTED] uses a single string to store all replies to a passage, with a single reply encased in  \' ... \' or " ... " if the reply contains a \'

    Args:
        rev (Optional[str]): Raw string containing all reviews (may be None)

    Returns:
        List[str]: list with each item a separate review.
    """
    if rev is None or len(rev) == 2:
        return []  # No reviews
    reviews = []
    match = None
    escape = False
    rev_single = ""  # Review to be added to list of reviews
    for char in rev[1:-1]:  # iterate with [] removed\
        if match is None:  # Starting a new review
            if char == '"' or char == "'":  # skips comma and space
                match = char
                rev_single = ""
            continue
        elif not escape and match == char:  # At the end of a review
            reviews.append(rev_single)
            match = None
        else:
            escape = False
            if char == "\\":
                escape = True
            if check_char(char):
                rev_single += char
    return reviews


def filter_empty(passages: List[str], reviews: List[str]) -> None:
    """Filters out passages with no reviews.
    Removes passages and reviews in-place if a passage has a
    Args:
        passages (List[str]): Short passages of text, collected in a list.
        reviews (List[str]): Reviews of that text, with the review at index `i` responding to `passages[i]`
    Raises:
        ValueError: The lists passages and reviews do not have the same length.
    """
    size = len(passages)
    if len(reviews) != size:
        raise ValueError(
            f"# of reviews {len(reviews)} != # of passages {size}. Check your data."
        )
    i = 0
    while i < size:
        if reviews[i] == "[]" or reviews[i] == []:
            del reviews[i]
            del passages[i]
            size -= 1
        else:
            i += 1


def create_tok(tokenizer: Callable, context_len: int):
    @typechecked
    def _tok(string_batch: Iterable[str]) -> BatchEncoding:
        for i, _ in enumerate(string_batch):
            if len(string_batch[i]) > context_len:
                string_batch[i] = string_batch[i][-context_len:]
        if not isinstance(string_batch, list):
            string_batch = list(string_batch)
        return tokenizer(string_batch)

    return _tok


@dataclass
class BatchElement:
    input_ids: TensorType[-1, "pass_N"]
    mask: TensorType[-1, "pass_N"]


# Assumes first axis of all tensor attributes in data are the same
# If no tensor attributes, returns original data object
def chunkBatchElement(data: BatchElement, chunk_size: int) -> List[BatchElement]:
    keys = list(vars(data).keys())
    n_keys = len(keys)
    is_tensor = []
    tensor_batch_dim = -1

    # Create data of same type as data
    data_class = type(data)

    # Mark which attrs are tensor types
    for key in keys:
        if torch.is_tensor(vars(data)[key]):
            is_tensor.append(True)
        else:
            is_tensor.append(False)

    # If no tensor type just return
    has_tensor = False
    for t in is_tensor:
        if t:
            has_tensor = True
            break
    if not has_tensor:
        return data

    # Check length of tensor type
    for is_t, key in zip(is_tensor, keys):
        if is_t:
            tensor_batch_dim = len(vars(data)[key])
            break

    # create indices for the chunks
    n_chunks = math.ceil(tensor_batch_dim / chunk_size)
    chunk_inds = torch.arange(tensor_batch_dim).chunk(n_chunks)

    # create new BatchElements that have chunks
    new_datas = []
    for inds in chunk_inds:
        data_args = []
        for is_t, key in zip(is_tensor, keys):
            if is_t:
                data_args.append(vars(data)[key][inds])
            else:
                data_args.append(vars(data)[key])

        new_datas.append(data_class(*data_args))

    return new_datas
