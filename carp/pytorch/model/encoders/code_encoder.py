from abc import abstractmethod
from typing import Iterable

import torch
import torch.nn.functional as F
from torchtyping import TensorType
from transformers import AutoModel, AutoTokenizer

from carp.pytorch.model.encoders import (
    BaseEncoder,
    BaseEncoderOutput,
    register_encoder,
)


@register_encoder
class SumCodeEncoder(BaseEncoder):
    def __init__(self, model_path: str, model_arch: str, tokenizer_path: str = None):
        super().__init__(model_path, model_arch, tokenizer_path, skip_init=True)
        self.model = AutoModel.from_pretrained(model_path)
        if tokenizer_path is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # add quote token to model and tokenizer
        strs_list = ["[START]", "[END]"]
        for idx in range(10):
            strs_list.append("[PLACEHOLDER-"+str(idx)+"]")
        self.tokenizer.add_tokens(strs_list)
        self.model.resize_token_embeddings(len(self.tokenizer))


    def preprocess(self, string_batch: Iterable[str]) -> Iterable[str]:
        return string_batch

    def forward(
        self,
        x,
        mask=None,
        tokenize: bool = False,
        mask_sum: bool = True,
        inputs_embeds: bool = False,
    ) -> TensorType["batch", "embed_dim"]:
        if tokenize:
            x = self.call_tokenizer(x)
            mask = x["attention_mask"]
            x = x["input_ids"]
            
        x = x[:, :512]
        mask = mask[:, :512]

        out = super().forward(x=x, attention_mask=mask, inputs_embeds=inputs_embeds)

        hidden: TensorType["batch", "N", "embed_dim"] = self.extract_fn(out)
        # Mask out pad tokens embeddings
        if mask_sum:
            emb_mask = mask.unsqueeze(2).repeat(1, 1, self.d_model)
            hidden = hidden * emb_mask
        return BaseEncoderOutput(F.normalize(hidden.sum(1)))  # Sum along sequence
