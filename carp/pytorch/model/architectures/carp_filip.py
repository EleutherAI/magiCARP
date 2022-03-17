import sys
sys.path.append('.')

from typing import List, Dict

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

    def _compute_loss_or_acc(
        self, 
        x, #: TensorType[-1, "latent_dim"], 
        y, #: TensorType[-1, "latent_dim"],
        normalize=False,
        logits_ij = None,
        return_loss = True,
        return_acc=True,
    ):
        n = x.shape[0]
        labels = torch.arange(n, device=self.config.device)
        
        if logits_ij is None:
            logits_ij = self.item_logits__mode_i_to_mode_j(x,y,normalize)

        outv = {}
        if return_loss:
            outv['loss'] = F.cross_entropy(logits_ij, labels)
        if return_acc:
            outv['acc'] = (torch.argmax(logits_ij, dim=1) == labels).sum() / n
        return outv


    def loss_component__mode_i_to_mode_j(
        self, 
        x, #: TensorType[-1, "latent_dim"], 
        y, #: TensorType[-1, "latent_dim"],
        normalize=False,
        logits_ij = None,
    ) -> TensorType[(), float]:

        d = self._compute_loss_or_acc(
            x=x,
            y=y,
            normalize=normalize,
            logits_ij = logits_ij,
            return_loss = True,
            return_acc=False,
        )
        return d['loss']


    def loss_component__mode_j_to_mode_i(
        self, 
        x, #: TensorType[-1, "latent_dim"], 
        y, #: TensorType[-1, "latent_dim"],
        normalize=False,
        logits_ji = None,
    ) -> TensorType[(), float]:
        return self.loss_component__mode_i_to_mode_j(y=y,x=x,normalize=normalize, logits_ij=logits_ji)


    def contrastive_loss_terms(
        self, 
        x, #: TensorType[-1, "latent_dim"], 
        y, #: TensorType[-1, "latent_dim"],
        normalize=False,
    ) -> TensorType[(), float]:

        loss_ij = self.loss_component__mode_i_to_mode_j(x,y,normalize)
        loss_ji = self.loss_component__mode_j_to_mode_i(x,y,normalize)
        return loss_ij, loss_ji

    def loss_components(
        self, 
        x=None,
        y=None,
        normalize=False,
        loss_ij=None,
        loss_ji=None,
        use_loss_transpose=True,
    ):
        if loss_ij is None:
            loss_ij = self.loss_component__mode_i_to_mode_j(x,y,normalize)
        if loss_ji is None:
            loss_ji = loss_ij.T
            if not use_loss_transpose:
                loss_ji = self.loss_component__mode_j_to_mode_i(x,y,normalize)
        return loss_ij, loss_ji

    def contrastive_loss(
        self, 
        x=None,
        y=None,
        normalize=False,
        loss_ij=None,
        loss_ji=None,
        use_loss_transpose=True,
    ) -> TensorType[(), float]:
        losses = self.loss_components(x=x,y=y, normalize=normalize, loss_ij=loss_ij, loss_ji=loss_ji, use_loss_transpose=use_loss_transpose)
        return sum(losses) / len(losses)

    def acc_component__mode_i_to_mode_j(
        self, 
        x, #: TensorType[-1, "latent_dim"], 
        y, #: TensorType[-1, "latent_dim"],
        normalize=False,
        logits_ij = None,
    ) -> TensorType[(), float]:

        d = self._compute_loss_or_acc(
            x=x,
            y=y,
            normalize=normalize,
            logits_ij = logits_ij,
            return_loss = False,
            return_acc=True,
        )
        return d['acc']

    def acc_component__mode_j_to_mode_i(
        self, 
        x, #: TensorType[-1, "latent_dim"], 
        y, #: TensorType[-1, "latent_dim"],
        normalize=False,
        logits_ji = None,
    ) -> TensorType[(), float]:
        return self.acc_component__mode_i_to_mode_j(y=y,x=x,normalize=normalize, logits_ij=logits_ji)

    def compute_accuracy(
        self, 
        x, #: TensorType[-1, "latent_dim"], 
        y, #: TensorType[-1, "latent_dim"],
        normalize=False,
        logits_ij = None,
        logits_ji = None,
    ):
        acc_ij = self.acc_component__mode_i_to_mode_j(x=x,y=y,normalize=normalize, logits_ij=logits_ij)
        acc_ji = self.acc_component__mode_j_to_mode_i(x=x,y=y,normalize=normalize, logits_ji=logits_ji)
        return (acc_ij + acc_ji) / 2


    # Is there a way we can inherit the typing from a method's output?
    def compute_accuracy_OLD(
        self, 
        x=None,
        y=None,
        normalize=False,
        loss_ij=None,
        loss_ji=None,
        use_loss_transpose=True,
    ) -> TensorType[(), float]:
        losses = self.loss_components(x=x,y=y, normalize=normalize, loss_ij=loss_ij, loss_ji=loss_ji, use_loss_transpose=use_loss_transpose)
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
class CARPFilipOLD(CARPSimRefactor):

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



@typechecked
@register_architecture
class CARPFilip(CARPFilipOLD):

    def forward(
        self,
        passages: BatchElement,
        reviews: BatchElement,
        config: TrainConfig,
    ):
        microbatch_inds = generate_indices(
            passages.input_ids.shape[0], config.microbatch_size, shuffle=False
        )
        # Split tokens and masks into these microbatches
        pass_mbs: List[BatchElement] = [
            BatchElement(passages.input_ids[i], passages.mask[i])
            for i in microbatch_inds
        ]
        rev_mbs: List[BatchElement] = [
            BatchElement(reviews.input_ids[i], reviews.mask[i]) for i in microbatch_inds
        ]

        # Initially get all encodings without grad
        pass_encs, rev_encs = self.calculate_embeddings(pass_mbs, rev_mbs)

        # compute accuracy
        forward_acc = self.compute_accuracy(torch.cat(pass_encs), torch.cat(rev_encs))

        return {
            "pass_mbs": pass_mbs,
            "pass_encs": pass_encs,
            "rev_mbs": rev_mbs,
            "rev_encs": rev_encs,
            "forward_acc": forward_acc,
        }

@register_trainer
class CARPFilipTrainer(CARPTrainer):
    def train_deepspeed_step(
        self,
        passages: BatchElement,
        reviews: BatchElement,
        config: TrainConfig,
    ):
        #forward_output = self.model(passages, reviews, config)

        microbatch_inds = generate_indices(
            passages.input_ids.shape[0], config.microbatch_size, shuffle=False
        )
        
        pass_mbs: List[BatchElement] = [
            BatchElement(passages.input_ids[i], passages.mask[i])
            for i in microbatch_inds
        ]

        with torch.no_grad():
            rev_encs = [self.model.encode_reviews(r).hidden for r in reviews]

        pass_temp = []
        for i in microbatch_inds:
            with torch.cuda.amp.autocast():
                passage = pass_mbs[i]
                pass_tmp_i = self.model.module.encode_passages(passage).hidden
                pass_temp.append(pass_tmp_i)
            loss = self.model.module.contrastive_loss(
                torch.cat(pass_tmp), torch.cat(rev_encs)
            )
            self.deepspeed_backwards(loss)
        rev_encs = None
        pass_mbs = None


        rev_mbs: List[BatchElement] = [
            BatchElement(reviews.input_ids[i], reviews.mask[i])
            for i in microbatch_inds
        ]

        with torch.no_grad():
            pass_encs = [self.model.encode_passages(p).hidden for p in passages]

        rev_temp = []
        for i in microbatch_inds:
            with torch.cuda.amp.autocast():
                review = rev_mbs[i]
                rev_tmp_i = self.model.module.encode_passages(review).hidden
                rev_temp.append(rev_tmp_i)
            loss = self.model.module.contrastive_loss(
                torch.cat(rev_tmp), torch.cat(pass_encs)
            )
            self.deepspeed_backwards(loss)
        pass_encs = None
        rev_mbs = None

        # Average the model gradients
        self.average_gradients()

        # Clipping
        self.clip_gradients()

        # Step the model
        self.deepspeed_step()

        return {
            "Loss/Train": loss,
            "Acc/Forward": forward_output["forward_acc"],
        }

    def train_torch_step(
        self,
        passages: BatchElement,
        reviews: BatchElement,
        config: TrainConfig,
    ) -> Dict[str, TensorType[()]]:
        forward_output = self.model(passages, reviews, config)

        # does gradient accumulation
        self.zero_grad()

        # Encode passages in microbatches (with grad)
        for index, passage in enumerate(forward_output["pass_mbs"]):
            pass_tmp = forward_output["pass_encs"].copy()
            with torch.cuda.amp.autocast():
                pass_tmp[index] = self.model.encode_passages(passage).hidden
                loss = self.model.contrastive_loss(
                    torch.cat(pass_tmp), torch.cat(forward_output["rev_encs"])
                )

            self.torch_backwards(loss)

        # Encode reviews in microbatches (with grad)
        for index, review in enumerate(forward_output["rev_mbs"]):
            rev_tmp = forward_output["rev_encs"].copy()  # no_grad
            with torch.cuda.amp.autocast():
                rev_tmp[index] = self.model.encode_reviews(review).hidden
                # grad _just_ at positions in `index`
                loss = self.model.contrastive_loss(
                    torch.cat(forward_output["pass_encs"]), torch.cat(rev_tmp)
                )

            self.torch_backwards(loss)

        # Average the model gradients
        self.average_gradients()

        # Clipping
        self.clip_gradients()

        # Step the model
        self.torch_step()

        return {
            "Loss/Train": loss,
            "Acc/Forward": forward_output["forward_acc"],
        }
