from unittest.mock import patch

import pytest
from transformers import AutoTokenizer

from carp.pytorch.data import BaseDataPipeline

toy_length = 2048
model_path = "haisongzhang/roberta-tiny-cased"
batch_size = 48


@pytest.fixture
def toy_dataset():
    train = {
        "story_target": ["a b c d e f g" for _ in range(toy_length)],
        "target_comment": ["h i j k l m n o" for _ in range(toy_length)],
    }
    data = {"train": train}
    return data


@pytest.fixture
def carp(toy_dataset):
    with patch("carp.pytorch.dataset.load_from_disk") as ds_patch:
        ds_patch.return_value = toy_dataset
        return BaseDataPipeline(dupe_protection=False)


def test_carpdataset(carp):
    assert len(carp) == toy_length


def test_tokenizer_factory(carp):
    hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = lambda x: hf_tokenizer(x, return_tensors="pt", padding=True)
    tok = BaseDataPipeline.tokenizer_factory(tokenizer, 20)
    data = [carp[i] for i in range(batch_size)]
    passages, reviews = tok(data)
    assert len(passages) == len(reviews)
    assert passages[0].shape[0] == reviews[0].shape[0]
    assert passages[0].shape[0] == batch_size
