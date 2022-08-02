from abc import abstractmethod
from typing import Iterable

import bitsandbytes as bnb
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
            strs_list.append("[PLACEHOLDER-" + str(idx) + "]")
        self.tokenizer.add_tokens(strs_list)
        self.model.resize_token_embeddings(len(self.tokenizer))
        # copy the model embedding layer to a bnb stable embedding layer
        embeddings = self.model.get_input_embeddings()
        self.model.set_input_embeddings(
            bnb.nn.StableEmbedding(len(self.tokenizer), self.model.config.hidden_size)
        )
        self.model.get_input_embeddings().weight.data.copy_(embeddings.weight.data)

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

        out = super().forward(x=x, attention_mask=mask, inputs_embeds=inputs_embeds)

        hidden: TensorType["batch", "N", "embed_dim"] = self.extract_fn(out)
        # Mask out pad tokens embeddings
        if mask_sum:
            emb_mask = mask.unsqueeze(2).repeat(1, 1, self.d_model)
            hidden = hidden * emb_mask
        return BaseEncoderOutput(F.normalize(hidden.sum(1)))  # Sum along sequence


@register_encoder
class MeanPoolCodeEncoder(BaseEncoder):
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
            strs_list.append("[PLACEHOLDER-" + str(idx) + "]")
        self.tokenizer.add_tokens(strs_list)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def preprocess(self, string_batch: Iterable[str]) -> Iterable[str]:
        return string_batch

    def mean_pooling(self, model_output, attention_mask):
        """
        Uses the attention mask to perform mean pooling over hidden states
        Args:
            model_output: Encoder hidden state ~ [bs, seq_len, hidden]
            attention_mask: Attention mask ~ [bs, seq_len]
        Returns:
            Mean pool ~ [bs, hidden]
        """
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(
        self, x, mask=None, inputs_embeds=False, **kwargs
    ) -> TensorType["batch", "embed_dim"]:
        out = super().forward(
            x=x, attention_mask=mask, inputs_embeds=inputs_embeds, **kwargs
        )
        return BaseEncoderOutput(self.mean_pooling(out, mask))


@register_encoder
class CausalMeanPoolCodeEncoder(MeanPoolCodeEncoder):
    def __init__(self, model_path: str, model_arch: str, tokenizer_path: str = None):
        super().__init__(model_path, model_arch, tokenizer_path)

        # add quote token to model and tokenizer
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(
        self, x, mask=None, inputs_embeds=False
    ) -> TensorType["batch", "embed_dim"]:
        return super().forward(
            x=x, mask=mask, inputs_embeds=inputs_embeds, use_cache=False
        )
