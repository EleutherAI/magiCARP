from abc import abstractmethod
from dataclasses import dataclass
from typing import Iterable, Optional
from torch import nn
import torch.nn.functional as F
from torchtyping import TensorType

from transformers import RobertaForMaskedLM, AutoTokenizer, AutoConfig
from carp.pytorch.model.encoders import register_encoder, BaseEncoder, BaseEncoderOutput

@dataclass
class MLMEncoderOutput(BaseEncoderOutput):
    loss : Optional[TensorType[()]]

# Abstract base for a model that can serve both for MLM and encoding
# Makes assumption that central model is MLM (i.e. it's roberta)
class MLMEncoder(BaseEncoder):

    def __init__(self, model_path: str, model_arch: str):
        super().__init__(model_path=model_path, model_arch=model_arch, skip_init=True)
        self.model = RobertaForMaskedLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # default roberta extract doesnt work
        self.extract_fn = lambda out: out['hidden_states'][-1]

        self.cfg = AutoConfig.from_pretrained(model_path)
        self.d_model = self.cfg.hidden_size
        
        # add quote token to model and tokenizer
        self.tokenizer.add_tokens(["[quote]"])
        self.model.resize_token_embeddings(len(self.tokenizer))

    @property
    def device(self):
        return self.model.device

    @abstractmethod
    def preprocess(self, string_batch: Iterable[str]) -> Iterable[str]:
        pass

    @abstractmethod
    def process_hidden_state(self, hidden: TensorType['batch', 'N', 'embed_dim'], mask: TensorType['batch', 'N']) -> TensorType['batch', 'embed_dim']:
        pass

    def tok(self, string_batch: Iterable[str]):
        return self.tokenizer(
            self.preprocess(string_batch),
            return_tensors='pt',  # Will they ever _not_ be pytorch tensors?
            padding=True,
        )

    def forward(self, x, mask = None, labels = None):
        out = {}
        if not labels is None:
            model_output = self.model(
                    input_ids = x, 
                    attention_mask = mask,
                    labels = labels,
                    output_hidden_states = True
            )
            loss =  model_output['loss']
        else:
            model_output = self.model(
                    input_ids = x, 
                    attention_mask = mask,
                    output_hidden_states = True
            )
            loss = None

        hidden: TensorType['batch', 'N', 'embed_dim'] = self.extract_fn(model_output)
        return MLMEncoderOutput(self.process_hidden_state(hidden, mask), loss)


# Same as summed text but alternates as being an MLM
@register_encoder
class MLMSumEncoder(MLMEncoder):

    def __init__(self, model_path: str, model_arch: str):
        super().__init__(model_path, model_arch)

    def preprocess(self, string_batch: Iterable[str]) -> Iterable[str]:
        return string_batch

    def process_hidden_state(self,
        hidden: TensorType['batch', 'N', 'embed_dim'],
        mask: TensorType['batch', 'N']) -> TensorType['batch', 'embed_dim']:
        if mask != None:
            emb_mask = mask.unsqueeze(2).repeat(1, 1, self.d_model)
            hidden = hidden * emb_mask
        return F.normalize(hidden.sum(1))