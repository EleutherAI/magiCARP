from abc import abstractmethod
from typing import Iterable, List

import torch
import torch.nn.functional as F
from torchtyping import TensorType
from transformers import (
    AutoModel,
    AutoTokenizer,
    PretrainedBartModel,
    PreTrainedModel,
)

from carp.pytorch.model.encoders import (
    BaseEncoder,
    BaseEncoderOutput,
    register_encoder,
)


class SharedEncoder(BaseEncoder):
    def __init__(
        self,
        model_path: str,
        model_arch: str,
        tokenizer_path: str = None,
        model: PreTrainedModel = None,
        is_review_encoder: bool = False,
    ):

        super().__init__(model_path=model_path, model_arch=model_arch, skip_init=True)

        self.is_review_encoder = is_review_encoder
        # each component of the shared encoder will have its own tokenizer
        # since too much of the data pipeline depends on this
        if tokenizer_path is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # add quote token to model and tokenizer
        self.tokenizer.add_tokens(["[quote]", "[story]", "[critique]"])

        # if the model was not already initialized on the other encoder
        # initialize it here and resize the embeddings
        if model is None:
            self.model = AutoModel.from_pretrained(model_path)
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.model = model

    # adds corresponding prefixes to tell the encoder which modality we are referring to
    def add_prefix(self, text: str):
        if self.is_review_encoder:
            return "[critique]" + text
        else:
            return "[story]" + text

    def preprocess(self, string_batch: Iterable[str]) -> Iterable[str]:
        return [self.add_prefix(s) for s in string_batch]


@register_encoder
class SharedSumTextEncoder(SharedEncoder):
    def __init__(
        self,
        model_path: str,
        model_arch: str,
        tokenizer_path: str = None,
        model: PreTrainedModel = None,
        is_review_encoder: bool = False,
    ):

        super().__init__(
            model_path, model_arch, tokenizer_path, model, is_review_encoder
        )

    def preprocess(self, string_batch: Iterable[str]) -> Iterable[str]:
        return super().preprocess(string_batch)

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
class SharedEOTTextEncoder(SharedEncoder):
    def __init__(
        self,
        model_path: str,
        model_arch: str,
        tokenizer_path: str = None,
        model: PreTrainedModel = None,
        is_review_encoder: bool = False,
    ):

        super().__init__(
            model_path, model_arch, tokenizer_path, model, is_review_encoder
        )

        # Add eot,pad token to model and tokenizer
        self.tokenizer.add_tokens(["<|endoftext|>"])
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.model.resize_token_embeddings(len(self.tokenizer))

    def preprocess(self, string_batch: Iterable[str]) -> Iterable[str]:
        """Adds end-of-text token (`<|endoftext|>`) to the end of each string in batch.

        Args:
            string_batch (Iterable[str]): The batch of text that will be encoded.

        Returns:
            List[str]: list of modified strings
        """
        return super().preprocess([s + "<|endoftext|>" for s in string_batch])

    def forward(self, x, mask=None, inputs_embeds=False):
        out = super().forward(x=x, attention_mask=mask, inputs_embeds=inputs_embeds)

        hidden = self.extract_fn(out)
        eot_inds = self.last_ones(mask)
        return BaseEncoderOutput(hidden[torch.arange(hidden.size(0)), eot_inds])


# Adds CLS token to start of string, end of string and middle of string
@register_encoder
class SharedMultiCLSEncoder(SharedEncoder):
    def __init__(
        self,
        model_path: str,
        model_arch: str,
        tokenizer_path: str = None,
        model: PreTrainedModel = None,
        is_review_encoder: bool = False,
    ):

        super().__init__(
            model_path, model_arch, tokenizer_path, model, is_review_encoder
        )

        super().__init__(model_path, model_arch)
        self.tokenizer.add_tokens(["[CLS]"])
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.model.resize_token_embeddings(len(self.tokenizer))

    @abstractmethod
    def add_cls(text: str) -> str:
        text = f"[CLS] {text} [CLS]"
        return text[: len(text) // 2] + "[CLS]" + text[len(text) // 2 :]

    def preprocess(self, string_batch: Iterable[str]) -> Iterable[str]:
        """Adds multiple class tokens (`[CLS]`) to the strings that will be encoded.

        Args:
            string_batch (Iterable[str]): Batch of strings

        Returns:
            List[str]: list of modified strings
        """
        return super().preprocess([self.add_cls(s) for s in string_batch])

    def forward(
        self, x, mask=None, inputs_embeds=False
    ) -> TensorType["batch", "embed_dim"]:
        out = super().forward(x=x, attention_mask=mask, inputs_embeds=inputs_embeds)
        hidden: TensorType["batch", "N", "embed_dim"] = self.extract_fn(out)
        batch_size = hidden.size(0)
        # start_inds are just 0-th position
        end_inds = self.last_ones(mask)
        mid_inds = end_inds // 2
        start_embed = hidden[:, 0]
        mid_embed = hidden[torch.arange(batch_size), mid_inds]
        end_embed = hidden[torch.arange(batch_size), end_inds]
        return BaseEncoderOutput(F.normalize(start_embed + mid_embed + end_embed))


@register_encoder
class SharedDirectTextEncoder(SharedEncoder):
    def __init__(
        self,
        model_path: str,
        model_arch: str,
        tokenizer_path: str = None,
        model: PreTrainedModel = None,
        is_review_encoder: bool = False,
    ):

        super().__init__(
            model_path, model_arch, tokenizer_path, model, is_review_encoder
        )

    def preprocess(self, string_batch: Iterable[str]) -> Iterable[str]:
        return super().preprocess(string_batch)

    def forward(self, x, mask=None, tokenize: bool = False):
        if tokenize:
            x = self.call_tokenizer(x)
            mask = x["attention_mask"]
            x = x["input_ids"]
        out = self.model(input_ids=x, attention_mask=mask)[0]
        embed = torch.sum(out * mask.unsqueeze(-1), dim=1) / torch.clamp(
            torch.sum(mask, dim=1, keepdims=True), min=1e-9
        )
        return BaseEncoderOutput(embed)


@register_encoder
class SharedMeanPoolEncoder(SharedEncoder):
    def __init__(
        self,
        model_path: str,
        model_arch: str,
        tokenizer_path: str = None,
        model: PreTrainedModel = None,
        is_review_encoder: bool = False,
    ):

        super().__init__(
            model_path, model_arch, tokenizer_path, model, is_review_encoder
        )

    def preprocess(self, string_batch: Iterable[str]) -> Iterable[str]:
        return super().preprocess(string_batch)

    def mean_pooling(self, model_output, attention_mask):
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
        out = super().forward(x=x, attention_mask=mask, inputs_embeds=inputs_embeds, **kwargs)
        return BaseEncoderOutput(self.mean_pooling(out, mask))


@register_encoder
class CausalSharedMeanPoolEncoder(SharedMeanPoolEncoder):
    def __init__(
        self,
        model_path: str,
        model_arch: str,
        tokenizer_path: str = None,
        model: PreTrainedModel = None,
        is_review_encoder: bool = False,
    ):
        super().__init__(model_path, model_arch, tokenizer_path, model, is_review_encoder)

        # add quote token to model and tokenizer
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(
        self, x, mask=None, inputs_embeds=False
    ) -> TensorType["batch", "embed_dim"]:
        return super().forward(
            x=x, mask=mask, inputs_embeds=inputs_embeds, use_cache=False
        )
