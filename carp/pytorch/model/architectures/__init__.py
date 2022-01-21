from __future__ import annotations

import sys
from typing import Dict, Tuple, Iterable, List
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

import torch
import torch.nn.functional as F
from torch import nn

from carp.pytorch.data.utils.data_util import BatchElement, chunkBatchElement
from carp.configs import TrainConfig
from carp.pytorch.model.encoders import get_encoder

# specifies a dictionary of architectures
_ARCHITECTURES: Dict[str, any] = {}  # registry

def register_architecture(name):
    """Decorator used register a CARP architecture 

        Args:
            name: Name of the architecture
    """

    def register_class(cls, name):
        _ARCHITECTURES[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)
    
    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls


patch_typeguard()

@typechecked
class BaseModel(nn.Module):
    """Abstract class that defines the basic API used for the different contrastive models."""

    def __init__(self, config, skip_init=False):
        super().__init__()
        if not skip_init:
            self.config = config
            encoder_class = get_encoder(config.encoder_type)
            self.passage_encoder = encoder_class(
                config.model_path, config.model_arch
            )
            self.review_encoder = encoder_class(
                config.model_path, config.model_arch
            )
            self.latent_dim = self.config.latent_dim
            self.pass_projector, self.rev_projector = self._make_projection_layers(self.config)
            self.logit_scale = nn.Parameter(
                torch.ones([], device=self.config.device)
                * torch.log(torch.tensor([1 / 0.07], device=self.config.device))
            )
            self.clamp_min = torch.log(torch.tensor([1 / 100], device=self.config.device))
            self.clamp_max = torch.log(torch.tensor([100], device=self.config.device))
        # used to count the number of steps until the next accumulation 
        self.accum_step = 0
        self.config = config

    # saves the model to the output directory. saved in chunks so that config can be swapped later
    def save(self, path : str):
        torch.save(self.passage_encoder.model, path + "passage_encoder.pt")
        torch.save(self.review_encoder.model, path + "review_encoder.pt")

        torch.save(self.pass_projector, path + "pass_projector.pt")
        torch.save(self.rev_projector, path + "rev_projector.pt")

        torch.save(self.logit_scale, path + "logit_scale.pt")

    # must be run after initialize 
    def load(self, path : str):
        self.passage_encoder.model = torch.load(path + "passage_encoder.pt")
        self.review_encoder.model = torch.load(path + "review_encoder.pt")

        self.pass_projector = torch.load(path + "pass_projector.pt")
        self.rev_projector = torch.load(path + "rev_projector.pt")

        self.logit_scale = torch.load(path + "logit_scale.pt")

    def compute_accuracy(self, x: TensorType[-1, "latent_dim"], y: TensorType[-1, "latent_dim"]):
        with torch.no_grad():
            n = x.shape[0]
            x = F.normalize(x)
            y = F.normalize(y)
            logits = x @ y.T * self.logit_scale.exp()
            labels = torch.arange(n, device=self.device)
            acc_i = (torch.argmax(logits, dim=1) == labels).sum()
            acc_t = (torch.argmax(logits, dim=0) == labels).sum()
        return (acc_i + acc_t) / n / 2

    def contrastive_loss(
        self, x: TensorType[-1, "latent_dim"], y: TensorType[-1, "latent_dim"]
    ) -> TensorType[(), float]:
        n = x.shape[0]
        x = F.normalize(x)
        y = F.normalize(y)
        # small term added to avoid nans in low precision softmax
        logits = (x @ y.T + 1e-6) * self.logit_scale.exp()
        labels = torch.arange(n, device=self.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_t) / 2
        
    def clamp(self):
        with torch.no_grad():
            self.logit_scale.clamp(self.clamp_min, self.clamp_max)

    @property
    def device(self):
        return self.passage_encoder.device

    def _make_projection_layers(self, config):
        if config.linear_projection:
            proj_pass = nn.Linear(
                self.passage_encoder.d_model, self.latent_dim, bias=False
            )
            proj_rev = nn.Linear(
                self.review_encoder.d_model, self.latent_dim, bias=False
            )
        else:
            proj_pass = Projection(
                self.passage_encoder.d_model, self.latent_dim, config.proj_dropout
            )
            proj_rev = Projection(
                self.review_encoder.d_model, self.latent_dim, config.proj_dropout
            )
        return proj_pass, proj_rev
            
    def _embed_data(
        self,
        x: BatchElement, 
        encoder,
        projector,
    ):
        x = encoder(x.input_ids.to(self.device), x.mask.to(self.device))
        x.hidden = projector(x.hidden)
        return x

    def encode_reviews(self, x):
        return self._embed_data(x, self.review_encoder, self.rev_projector)

    def encode_passages(self, x):
        return self._embed_data(x, self.passage_encoder, self.pass_projector)

    def calculate_embeddings(
        self,
        passages: Iterable[
            Tuple[
                BatchElement
            ]
        ],
        reviews: Iterable[
            Tuple[
                BatchElement
            ]
        ],
        return_only_embeddings : bool = True,
    ):
        # Get encodings without grad
        with torch.no_grad(), torch.cuda.amp.autocast():
            pass_encs = [self.encode_passages(p) for p in passages]
            rev_encs = [self.encode_reviews(r) for r in reviews]
        
        # if we only need the embeddings, fetch them
        if return_only_embeddings:
            pass_encs = list(map(lambda x: x.hidden, pass_encs))
            rev_encs = list(map(lambda x: x.hidden, rev_encs))
        return pass_encs, rev_encs

    def train_step(
        self,
        passages: BatchElement,
        reviews: BatchElement,
        config: TrainConfig,
        opt: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
    ) -> Dict[str, TensorType[()]]:
        raise NotImplementedError("Must be overridden.")

    # used to account for gradient accumulations
    def zero_grad(self,
        opt : torch.optim.Optimizer):
        if self.accum_step % self.config.grad_accum == 0:
            opt.zero_grad()
            
    def step(self,
        scaler : torch.cuda.amp.GradScaler,
        opt: torch.optim.Optimizer):
        if self.accum_step % self.config.grad_accum == 0:
            scaler.step(opt)
            scaler.update()
            self.accum_step = 0
        else:
            self.accum_step += 1


    def eval_step(self, dataset):
        passages = []
        reviews = []
        for p, r in dataset:
            passages.append(p)
            reviews.append(r)
        
        # TODO: Ideally should get microbatch size from trainconfig for the second argument
        passages = chunkBatchElement(passages[0], 8)
        reviews = chunkBatchElement(reviews[0], 8)

        with torch.no_grad():
            pass_emb, rev_emb = self.calculate_embeddings(passages, reviews)
            val_loss = self.contrastive_loss(torch.cat(pass_emb), torch.cat(rev_emb))
            val_acc = self.compute_accuracy(torch.cat(pass_emb), torch.cat(rev_emb))

        return {"Loss/Validation": val_loss.item(), "Acc/Validation": val_acc.item()}


# Project encoder output to latent space
class Projection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(
        self, x: TensorType["batch_dim", "in_dim"]
    ) -> TensorType["batch_dim", "out_dim"]:
        projected = self.proj(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        return self.layer_norm(x)

from carp.pytorch.model.architectures.carp import CARP
from carp.pytorch.model.architectures.carp_momentum import CARPMomentum
from carp.pytorch.model.architectures.carp_cloob import CARPCloob
from carp.pytorch.model.architectures.carp_mlm import CARPMLM
from carp.pytorch.model.architectures.carp_coop import CARPCoOp

def get_architecture(name):
    return _ARCHITECTURES[name.lower()]

def get_architecture_names():
    return _ARCHITECTURES.keys()
