import math
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers import AutoModel, AutoTokenizer

LATENT_DIM = 2048
USE_CUDA = True
USE_HALF = True
config = transformers.RobertaConfig()

extract_fns = {
    "EleutherAI/gpt-neo-1.3B": (lambda out: out["hidden_states"][-1]),
    "EleutherAI/gpt-neo-2.7B": (lambda out: out["hidden_states"][-1]),
    "roberta-large": (lambda out: out[0]),
    "roberta-base": (lambda out: out[0]),
    "microsoft/deberta-v2-xlarge": (lambda out: out[0]),
}

d_models = {
    "EleutherAI/gpt-neo-1.3B": 2048,
    "EleutherAI/gpt-neo-2.7B": 2560,
    "roberta-large": 1024,
    "roberta-base": 768,
    "microsoft/deberta-v2-xlarge": 1536,
}

MODEL_PATH = "roberta-large"


class TextEncoderV1(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = AutoModel.from_pretrained(MODEL_PATH)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.d_model = d_models[MODEL_PATH]

        # Add cls token to model and tokenizer
        self.tokenizer.add_tokens(["[quote]"])
        self.model.resize_token_embeddings(len(self.tokenizer))

    def tok(self, string_batch):
        return self.tokenizer(string_batch, return_tensors="pt", padding=True).to(
            "cuda"
        )

    def forward(self, x, mask=None, tokenize=False, mask_sum=True):
        if tokenize:
            x = self.tok(x)
            mask = x["attention_mask"]
            x = x["input_ids"]

        out = self.model(x, mask, output_hidden_states=True, return_dict=True)

        # out is a tuple of (model output, tuple)
        # the second tuple is all layers
        # in this second tuple, last elem is model output
        # we take second last hidden -> third last layer
        # size is always [batch, seq, 1536]

        hidden = out[0]
        # layers = out[-1]
        # hidden = layers[-2]

        # Mask out pad tokens embeddings
        if mask_sum:
            emb_mask = mask.unsqueeze(2).repeat(1, 1, self.d_model)
            hidden = hidden * emb_mask

        y = hidden.sum(1)
        y = F.normalize(y)

        return y  # Sum along sequence


class ContrastiveModelV1(nn.Module):
    def __init__(self, encA, encB):
        super().__init__()

        self.encA = encA
        self.encB = encB

        self.projA = nn.Linear(self.encA.d_model, LATENT_DIM, bias=False)
        self.projB = nn.Linear(self.encB.d_model, LATENT_DIM, bias=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.clamp_min = math.log(1 / 100)
        self.clamp_max = math.log(100)

    def clamp(self):
        """ "
        Clamps the temperature to stay within a certain range, as specified in OpenAI's original
        CLIP paper.
        """
        with torch.no_grad():
            self.logit_scale.clamp(self.clamp_min, self.clamp_max)

    def encodeX(self, strs: List[str], masks=None):
        """ "
        Embeds a text sequence using the passge embedder
        and sums along the sequence dimension (Optionally using mask)

        Args:
            strs: List of strings to be encoded and projected
            masks: Used to mask out tokens (For EOT classifiers)
        returns:
            Tensor of shape [bs, hidden_sim]
        """
        x = self.encA(strs, masks)
        return self.projA(x)

    def encodeY(self, strs: List[str], masks=None):
        """ "
        Embeds a text sequence using the critique embedder
        and sums along the sequence dimension (Optionally using mask)

        Args:
            strs: List of strings to be encoded and projected
            masks: Used to mask out tokens (For EOT classifiers)
        returns:
            Tensor of shape [bs, hidden_sim]
        """
        y = self.encB(strs, masks)
        return self.projB(y)

    # Calculate contrastive loss between embedding groups
    # x, y are assumed encoding/embeddings here
    def cLoss(self, x, y):
        """ "
        Computes contrastive loss given the embedded passages and critiques.
        Utilized within contrastive parallel

        Args:
            x: Tensor of embedded passages of shape [batch size, hidden_dim]
            y: Tensor of embedded critiques of shape [batch size, hidden_dim]
        returns:
            loss: Float
        """
        n = x.shape[0]
        # normalize
        x = F.normalize(x)
        y = F.normalize(y)

        logits = x @ y.T * self.logit_scale.exp()
        labels = torch.arange(n, device="cuda")

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        acc_i = (torch.argmax(logits, dim=1) == labels).sum()
        acc_t = (torch.argmax(logits, dim=0) == labels).sum()

        return (loss_i + loss_t) / 2, (acc_i + acc_t) / n / 2

    def getLogits(self, x, y):
        """ "
        Computes logits given the embedded passages and critiques.
        Utilized for inference

        Args:
            x: Tensor of embedded passages of shape [batch size, hidden_dim]
            y: Tensor of embedded critiques of shape [batch size, hidden_dim]
        returns:
            logits: Tensor of shape [batch size x, batch size y]
        """
        x = self.encodeX(*x)
        y = self.encodeY(*y)

        x = F.normalize(x)
        y = F.normalize(y)

        logits = x @ y.T * self.logit_scale.exp()
        return logits

    def forward(self, x, y):
        """ "
        Calls compute logits

        Args:
            x: Tensor of embedded passages of shape [batch size, hidden_dim]
            y: Tensor of embedded critiques of shape [batch size, hidden_dim]
        returns:
            logits: Tensor of shape [batch size x, batch size y]
        """
        return self.getLogits(x, y)
