from abc import abstractmethod
from typing import Iterable

import torch
import torch.nn.functional as F
from torchtyping import TensorType

from carp.pytorch.model.encoders import (
    BaseEncoder,
    BaseEncoderOutput,
    register_encoder,
)


@register_encoder
class NaiveTextEncoder(BaseEncoder):
    #def __init__(self, model_path: str, model_arch: str):
    #    super().__init__(model_path, model_arch)

    def preprocess(self, string_batch: Iterable[str]) -> Iterable[str]:
        return string_batch

    def forward(
        self,
        x,
        mask=None,
        tokenize: bool = False,
        inputs_embeds: bool = False,
    ) -> TensorType["batch", "embed_dim"]:
        if tokenize:
            x = self.call_tokenizer(x)
            mask = x["attention_mask"]
            x = x["input_ids"]
        out = super().forward(x=x, attention_mask=mask, inputs_embeds=inputs_embeds)

        embedded_token_sequence: TensorType["batch", "N", "embed_dim"] = self.extract_fn(out)
        return BaseEncoderOutput(embedded_token_sequence)
