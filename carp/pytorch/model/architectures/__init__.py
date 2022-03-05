from __future__ import annotations

import sys
from abc import abstractmethod
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn, no_grad
from torch.cuda.amp import autocast
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from carp.configs import TrainConfig
from carp.pytorch.data.utils.data_util import BatchElement, chunkBatchElement
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
                config.model_path, config.model_arch, config.tokenizer_path
            )
            self.review_encoder = encoder_class(
                config.model_path, config.model_arch, config.tokenizer_path
            )
            self.latent_dim = self.config.latent_dim
            self.pass_projector, self.rev_projector = self._make_projection_layers(
                self.config
            )
            self.logit_scale = nn.Parameter(
                torch.ones([], device=self.config.device)
                * torch.log(torch.tensor([1 / 0.07], device=self.config.device))
            )
            self.logit_scale.requires_grad = False
            self.clamp_min = torch.log(
                torch.tensor([1 / 100], device=self.config.device)
            )
            self.clamp_max = torch.log(torch.tensor([100], device=self.config.device))
        # used to count the number of steps until the next accumulation
        self.accum_step = 0
        self.config = config

    @abstractmethod
    def attempt_save(cls, component, path: str, component_name: str):
        """
        Attempts to save a component of the model. Throws an exception and continues if the component cannot be saved
        Args:
            component : Component to be saved using torch.save
            path : directory to save to
            component_name : name of component to append onto path
        """
        try:
            torch.save(component, path + component_name)
        except:
            print("Unable to save " + component_name + ". Continuing.")

    @abstractmethod
    def attempt_load(cls, path: str, component_name: str):
        """
        Attempts to load a component of the model. Throws an exception and continues if the component cannot be loaded
        Args:
            path : directory to load from
            component_name : name of component to append onto path
        Returns:
            component : nn.module
        """
        try:
            # kevin) cpu load -> gpu upload is better.
            return torch.load(path + component_name, map_location="cpu")
        except:
            print("Unable to load " + component_name + ". Continuing.")

    # saves the model to the output directory. saved in chunks so that config can be swapped later
    def save(self, path: str):
        self.attempt_save(self.passage_encoder.model, path, "passage_encoder.pt")
        self.attempt_save(self.review_encoder.model, path, "review_encoder.pt")

        self.attempt_save(self.pass_projector, path, "pass_projector.pt")
        self.attempt_save(self.rev_projector, path, "rev_projector.pt")
        try:
            self.attempt_save(self.logit_scale, path, "logit_scale.pt")
        except:
            pass

    # must be run after initialize
    def load(self, path: str):
        self.passage_encoder.model = self.attempt_load(path, "passage_encoder.pt")
        self.review_encoder.model = self.attempt_load(path, "review_encoder.pt")

        self.pass_projector = self.attempt_load(path, "pass_projector.pt")
        self.rev_projector = self.attempt_load(path, "rev_projector.pt")

        self.logit_scale = self.attempt_load(path, "logit_scale.pt")

    def compute_accuracy(
        self,
        x: TensorType[-1, "latent_dim"],
        y: TensorType[-1, "latent_dim"],
        normalize: bool = False,
    ):
        with no_grad():
            n = x.shape[0]
            if normalize:
                x = F.normalize(x)
                y = F.normalize(y)
            logits = x @ y.T * self.logit_scale.exp()
            labels = torch.arange(n, device=self.config.device)
            acc_i = (torch.argmax(logits, dim=1) == labels).sum()
            acc_t = (torch.argmax(logits, dim=0) == labels).sum()
        return (acc_i + acc_t) / n / 2

    def cosine_sim(
        self,
        x: TensorType[-1, "latent_dim"],
        y: TensorType[-1, "latent_dim"],
        normalize=False,
    ):
        """
        Computes the cosine similarity between two sets of vectors x,y
        Args:
            x: Tensor of passage embeddings
            y: Tensor of review embeddings
        Returns:
            Matrix of size pass_N x rev_N
        """
        if normalize:
            x = F.normalize(x)
            y = F.normalize(y)
        # small term added to avoid nans in low precision softmax
        return torch.abs(x @ y.T) + 1e-6

    def contrastive_loss(
        self, x: TensorType[-1, "latent_dim"], y: TensorType[-1, "latent_dim"]
    ) -> TensorType[(), float]:

        n = x.shape[0]
        # small term added to avoid nans in low precision softmax
        logits = self.cosine_sim(x, y) * self.logit_scale.exp()
        labels = torch.arange(n, device=self.config.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_t) / 2

    def clamp(self):
        with no_grad():
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
        normalize=False,
    ):
        x = encoder(x.input_ids.to(self.config.device), x.mask.to(self.config.device))
        x.hidden = projector(x.hidden)
        if normalize:
            x.hidden = F.normalize(x.hidden)
        return x

    def encode_reviews(self, x, normalize=True):
        return self._embed_data(x, self.review_encoder, self.rev_projector, normalize)

    def encode_passages(self, x, normalize=True):
        return self._embed_data(x, self.passage_encoder, self.pass_projector, normalize)

    def calculate_embeddings(
        self,
        passages: Iterable[Tuple[BatchElement]],
        reviews: Iterable[Tuple[BatchElement]],
        return_only_embeddings: bool = True,
    ):
        # Get encodings without grad
        with no_grad():
            pass_encs = [self.encode_passages(p) for p in passages]
            rev_encs = [self.encode_reviews(r) for r in reviews]

        # if we only need the embeddings, fetch them
        if return_only_embeddings:
            pass_encs = list(map(lambda x: x.hidden, pass_encs))
            rev_encs = list(map(lambda x: x.hidden, rev_encs))
        return pass_encs, rev_encs

    def forward(
        self,
        passages: BatchElement,
        reviews: BatchElement,
        config: TrainConfig,
    ) -> Dict[str, TensorType[()]]:
        raise NotImplementedError("Must be overridden.")


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
from carp.pytorch.model.architectures.carp_cloob import CARPCloob
from carp.pytorch.model.architectures.carp_coop import CARPCoOp
from carp.pytorch.model.architectures.carp_mlm import CARPMLM
from carp.pytorch.model.architectures.carp_momentum import CARPMomentum
from carp.pytorch.model.architectures.carp_shared_encoder import (
    CARPSharedEncoder,
)


def get_architecture(name):
    return _ARCHITECTURES[name.lower()]


def get_architecture_names():
    return _ARCHITECTURES.keys()
