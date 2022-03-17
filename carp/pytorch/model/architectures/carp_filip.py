import sys
sys.path.append('.')

from typing import List

import torch
import torch.nn.functional as F

from carp.configs import ModelConfig
#from carp.pytorch.model.architectures import *
from carp.pytorch.model.architectures import (
    #BaseModel,
    #Projection,
    #CARP,
    register_architecture,
    typechecked,
)
from carp.pytorch.model.architectures.carp import CARP, CARPTrainer
from carp.pytorch.training import BaseTrainer, register_trainer
from carp.util import generate_indices

from torchtyping import TensorType, patch_typeguard

# maybe useful?
from carp.pytorch.data.utils.data_util import BatchElement, chunkBatchElement
from carp.pytorch.model.encoders import get_encoder
from carp.configs import TrainConfig

import einops as eo
from loguru import logger

# TO DO: Make sure this behaves the same as CARP. 
#        If I didn't mess anything up, should be numerically identical.
@typechecked
@register_architecture
class CARPSimRefactor(CARP):
    """
    making the item-item similarity more general to facilitate cleaner FILIP implementation.
    """
    def item_pseudosimilarity__mode_i_to_mode_j(
        self,
        x: TensorType[-1, "latent_dim"],
        y: TensorType[-1, "latent_dim"],
        normalize=False,
    ):
        """
        "pseudo" similarity to permit asymmetric and mode-specifice simmilarity measures
        """
        return self.cosine_sim(x, y, normalize)

    def item_pseudosimilarity__mode_j_to_mode_i(
        self,
        y, #: TensorType[-1, "latent_dim"],
        x, #: TensorType[-1, "latent_dim"],
        normalize=False,
    ):
        """
        "pseudo" similarity to permit asymmetric and mode-specifice simmilarity measures
        """
        return self.item_pseudosimilarity__mode_i_to_mode_j(y,x, normalize)

    def item_logits__mode_i_to_mode_j(
        self,
        x: TensorType[-1, "latent_dim"],
        y: TensorType[-1, "latent_dim"],
        normalize=False,
    ) -> TensorType["batch_size", "batch_size"]:
        S_ij = self.item_pseudosimilarity__mode_i_to_mode_j(x,y,normalize)
        return S_ij * self.logit_scale.exp()
    
    def item_logits__mode_j_to_mode_i(
        self,
        y, #: TensorType[-1, "latent_dim"],
        x, #: TensorType[-1, "latent_dim"],
        normalize=False,
        ) -> TensorType["batch_size", "batch_size"]:
        return self.item_logits__mode_i_to_mode_j(y,x,normalize)

    def contrastive_loss(
        self, 
        x, #: TensorType[-1, "latent_dim"], 
        y, #: TensorType[-1, "latent_dim"],
        normalize=False,
    ) -> TensorType[(), float]:

        n = x.shape[0]
        labels = torch.arange(n, device=self.config.device)
        
        logits_ij = self.item_logits__mode_i_to_mode_j(x,y,normalize)
        logits_ji = self.item_logits__mode_j_to_mode_i(y,x,normalize)
        #logger.debug(logits_ij.shape) [batch_size batch_size]

        loss_ij = F.cross_entropy(logits_ij, labels)
        loss_ji = F.cross_entropy(logits_ji, labels)
        return (loss_ij + loss_ji) / 2

    # Is there a way we can inherit the typing from a method's output?
    def compute_accuracy(
        self,
        x, #: TensorType[-1, "latent_dim"],
        y, #: TensorType[-1, "latent_dim"],
        normalize: bool = False,
    ):
        with torch.no_grad():
            n = x.shape[0]
            labels = torch.arange(n, device=self.config.device)
            
            logits_ij = self.item_logits__mode_i_to_mode_j(x,y,normalize)
            logits_ji = self.item_logits__mode_j_to_mode_i(y,x,normalize)
            acc_ij = (torch.argmax(logits_ij, dim=1) == labels).sum()
            acc_ji = (torch.argmax(logits_ji, dim=1) == labels).sum()
        return (acc_ij + acc_ji) / n / 2

@typechecked
@register_architecture
class CARPFilip(CARPSimRefactor):

    def item_pseudosimilarity__mode_i_to_mode_j(
        self,
        x: TensorType["batch_size", -1, "latent_dim"],
        y: TensorType["batch_size", -1, "latent_dim"],
        normalize=True,
    ):
        # To do
        #logger.debug(x.shape) # [batch_size, N, latent_dim]
        #logger.debug(y.shape) # [batch_size, M, latent_dim]
        #raise NotImplementedError
        if normalize:
            x = F.normalize(x)
            y = F.normalize(y)
        y2 = eo.rearrange(y, 'b m d -> b d m')
        # fancy broadcasting
        x2 = x.unsqueeze(1)
        z = torch.matmul(x2,y2)
        return z

    def item_logits__mode_i_to_mode_j(
            self,
            x: TensorType["batch_size", -1, "latent_dim"],
            y: TensorType["batch_size", -1, "latent_dim"],
            normalize=True,
        ):
        #logger.debug(x.shape) # [batch_size, N, latent_dim]
        #logger.debug(y.shape)
        S_ij = self.item_pseudosimilarity__mode_i_to_mode_j(x,y,normalize)
        #logger.debug(S_ij.shape)
        logits_ij = S_ij * self.logit_scale.exp()
        #logger.debug(logits_ij.shape)
        return logits_ij.max(dim=-1).values.mean(dim=-1)


@register_trainer
class CARPFilipTrainer(CARPTrainer):
    pass