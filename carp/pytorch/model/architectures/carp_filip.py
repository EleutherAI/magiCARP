import sys
sys.path.append('.') # ugh... really?

from typing import List, Callable

import torch
from torch import nn
import torch.nn.functional as F

from carp.pytorch.model.architectures import (
    register_architecture,
    typechecked,
)
from carp.pytorch.model.architectures.carp import CARP, CARPTrainer
from carp.pytorch.training.trainer import register_trainer
from carp.util import generate_indices

from torchtyping import TensorType

from carp.pytorch.data.utils.data_util import BatchElement
from carp.configs import TrainConfig

import einops as eo
from loguru import logger

@typechecked
@register_architecture
class CARPSimRefactor(CARP):
    """
    This class is just a refactoring of the CARP base architecture to facilitate microbatching
    over similarity computations that aren't microbatched in the current base CARP. It should
    behave numerically equivalent to the unmodified parent class.
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
        x, #: TensorType[-1, "latent_dim"],
        y, #: TensorType[-1, "latent_dim"],
        normalize=False,
    ):
        """
        "pseudo" similarity to permit asymmetric and mode-specifice simmilarity measures
        """
        return self.item_pseudosimilarity__mode_i_to_mode_j(x=y,y=x, normalize=normalize)

    def item_logits__mode_i_to_mode_j(
        self,
        x: TensorType[-1, "latent_dim"],
        y: TensorType[-1, "latent_dim"],
        normalize=False,
    #) -> TensorType["batch_size", "batch_size"]:
    ) -> TensorType:
        S_ij = self.item_pseudosimilarity__mode_i_to_mode_j(x=x,y=y,normalize=normalize)
        return S_ij * self.logit_scale.exp()

    def item_logits__mode_j_to_mode_i(
        self,
        x, #: TensorType[-1, "latent_dim"],
        y, #: TensorType[-1, "latent_dim"],
        normalize=False,
        #) -> TensorType["batch_size", "batch_size"]:
        ) -> TensorType:
        return self.item_logits__mode_i_to_mode_j(x=y,y=x,normalize=normalize)

    def _compute_loss_or_acc(
        self,
        x, #: TensorType[-1, "latent_dim"],
        y, #: TensorType[-1, "latent_dim"],
        normalize=False,
        logits_ij = None,
        return_loss = True,
        return_acc=True,
    ):
        if isinstance(logits_ij, list):
            logits_ij = torch.cat(logits_ij)
        try:
            n = x.shape[0]
        except AttributeError:
            n = logits_ij.shape[-1] # NB: n should correspond to batch_size, not microbatch_size
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
        return self.loss_component__mode_i_to_mode_j(x=y,y=x,normalize=normalize, logits_ij=logits_ji)


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
        logits_ij=None,
        logits_ji=None,
    ):
        if loss_ij is None:
            loss_ij = self.loss_component__mode_i_to_mode_j(x=x,y=y,normalize=normalize, logits_ij=logits_ij)
        if loss_ji is None:
            loss_ji = loss_ij.T
            if not use_loss_transpose:
                loss_ji = self.loss_component__mode_j_to_mode_i(x=x,y=y,normalize=normalize, logits_ji=logits_ji)
        return loss_ij, loss_ji

    def contrastive_loss(
        self,
        x=None,
        y=None,
        normalize=False,
        loss_ij=None,
        loss_ji=None,
        use_loss_transpose=True,
        logits_ij=None,
        logits_ji=None,
    ) -> TensorType[(), float]:

        losses = self.loss_components(
            x=x,y=y, normalize=normalize,
            loss_ij=loss_ij, loss_ji=loss_ji, use_loss_transpose=use_loss_transpose,
            logits_ij=logits_ij,
            logits_ji=logits_ji
            )
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
        x=None, #: TensorType[-1, "latent_dim"],
        y=None, #: TensorType[-1, "latent_dim"],
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

            logits_ij = self.item_logits__mode_i_to_mode_j(x=x,y=y,normalize=normalize)
            logits_ji = self.item_logits__mode_j_to_mode_i(x=x,y=y,normalize=normalize)
            acc_ij = (torch.argmax(logits_ij, dim=1) == labels).sum()
            acc_ji = (torch.argmax(logits_ji, dim=1) == labels).sum()
        return (acc_ij + acc_ji) / n / 2


@typechecked
@register_architecture
@typechecked
@register_architecture
class CARPFilip(CARPSimRefactor):

    def item_pseudosimilarity__mode_i_to_mode_j(
        self,
        x: TensorType["microbatch_size", -1, "latent_dim"],
        y: TensorType["microbatch_size", -1, "latent_dim"],
        normalize=True,
    ):
        #return self.item_pseudosimilarity__mode_i_to_mode_j_matmul(x,y,normalize)
        return self.item_pseudosimilarity__mode_i_to_mode_j_einsum(x,y,normalize)


    # huge thanks to MicPie,
    # will likely incorporate learnings from lucidrains' x-clip shortly
    def item_pseudosimilarity__mode_i_to_mode_j_einsum(
        self,
        x: TensorType["microbatch_size", -1, "latent_dim"],
        y: TensorType["microbatch_size", -1, "latent_dim"],
        normalize=True,
    ):
        if normalize:
            x = F.normalize(x)
            y = F.normalize(y)
        return torch.einsum('xmd,ynd->xymn', x, y)

    # numerically equivalent to the einsum version,
    # but WAAAY more memory-consumptive.
    # Leaving this here to document for posterity.
    # Don't use this. It's awful.
    def item_pseudosimilarity__mode_i_to_mode_j_matmul(
        self,
        x: TensorType["microbatch_size", -1, "latent_dim"],
        y: TensorType["microbatch_size", -1, "latent_dim"],
        normalize=True,
    ):
        if normalize:
            x = F.normalize(x)
            y = F.normalize(y)
        y2 = eo.rearrange(y, 'b m d -> b d m')
        # fancy broadcasting
        x2 = x.unsqueeze(1)
        z = torch.matmul(x2,y2) # this op is very memory heavy
        return z


    def item_logits__mode_i_to_mode_j(
            self,
            x: TensorType["microbatch_size", -1, "latent_dim"],
            y: TensorType["microbatch_size", -1, "latent_dim"],
            normalize=True,
        ):
        S_ij = self.item_pseudosimilarity__mode_i_to_mode_j(x,y,normalize)
        logits_ij = S_ij * self.logit_scale.exp()
        return logits_ij.max(dim=-1).values.mean(dim=-1)


@register_trainer
class CARPSimRefactorTrainer(CARPTrainer):

    def microbatch_up_logits__mode_i_to_mode_j(
        self,
        mode_w_grad: BatchElement,
        mode_no_grad: BatchElement,
        encoder_w_grad: nn.Module,
        encoder_no_grad: nn.Module,
        logits_func: Callable,
        config: TrainConfig,
        logit_chunks=None,
        f_backwards=None,
    ):
        microbatch_inds = generate_indices(
            config.batch_size, #mode_w_grad.input_ids.shape[0],
            config.microbatch_size,
            shuffle=False
        )

        with torch.no_grad():
            #enc_no_grad = encoder_no_grad(mode_no_grad).hidden
            # ...still explodes, need to microbatch over this one too
            mbs_enc_no_grad = [
                encoder_no_grad(
                    BatchElement(mode_no_grad.input_ids[i], mode_no_grad.mask[i])
                ).hidden
                for i in microbatch_inds
            ]

        # just parsing into microbatches here
        mbs_w_grad: List[BatchElement] = [
            BatchElement(mode_w_grad.input_ids[i], mode_w_grad.mask[i])
            for i in microbatch_inds
        ]

        compute_loss = True
        if logit_chunks is None:
            logit_chunks = []
            compute_loss = False
        for i, item in enumerate(mbs_w_grad):
            with torch.cuda.amp.autocast():
                enc_i = encoder_w_grad(item).hidden

            logits_ij_list = [logits_func(enc_i, enc_no_grad)
                for enc_no_grad in mbs_enc_no_grad]

            logits_ij = torch.cat(logits_ij_list, dim=-1) # [microbatch_size batch_size]

            if compute_loss:
                temp_logits = logit_chunks.copy()
                temp_logits[i] = logits_ij
                logits_cat = torch.cat(temp_logits) # logits_cat.shape -> [batch_size batch_size]

                loss = self.model.contrastive_loss(logits_ij=logits_cat) # probably cheating a bit here just assuming we can use the transpose.....
                f_backwards(loss)
            else:
                logit_chunks.append(logits_ij)
        return logit_chunks

    def _inner_step(self,
        mode_w_grad: BatchElement,
        mode_no_grad: BatchElement,
        config: TrainConfig,
        f_backwards=None,
    ):

        # 1. Build up logits_ij
        logger.debug("building logits_ij no grad")
        with torch.no_grad():
            logits_chunks_ij = self.microbatch_up_logits__mode_i_to_mode_j(
                mode_w_grad=mode_w_grad,
                mode_no_grad=mode_no_grad,
                encoder_w_grad=self.model.encode_passages,
                encoder_no_grad=self.model.encode_reviews,
                logits_func=self.model.item_logits__mode_i_to_mode_j,
                config=config,
                logit_chunks=None,
                f_backwards=None,
            )
        #logger.debug(type(logits_chunks_ij))
        #logger.debug(len(logits_chunks_ij))

        # 2. Ok, again but this time with grad and loss....
        logger.debug("building logits_ij with grad")
        logits_chunks_ij = self.microbatch_up_logits__mode_i_to_mode_j(
                mode_w_grad=mode_w_grad,
                mode_no_grad=mode_no_grad,
                encoder_w_grad=self.model.encode_passages,
                encoder_no_grad=self.model.encode_reviews,
                logits_func=self.model.item_logits__mode_i_to_mode_j,
                config=config,
                logit_chunks=logits_chunks_ij,
                f_backwards=f_backwards,
            )

        return logits_chunks_ij

    def train_deepspeed_step(
        self,
        passages: BatchElement,
        reviews: BatchElement,
        config: TrainConfig,
    ):
        with self.autocast():
            logits_ij = self._inner_step(
                mode_w_grad=passages,
                mode_no_grad=reviews,
                config=config,
                f_backwards= self.deepspeed_backwards,
            )

            logits_ji = self._inner_step(
                mode_w_grad=reviews,
                mode_no_grad=passages,
                config=config,
                f_backwards= self.deepspeed_backwards,
            )

        self.average_gradients()
        self.clip_gradients()
        self.deepspeed_step()

        with torch.no_grad():
            loss = self.model.module.contrastive_loss(logits_ij=logits_ij, logits_ji=logits_ji)
            acc = self.model.module.compute_accuracy(logits_ij=logits_ij, logits_ji=logits_ji)

        return {
            "Loss/Train": loss,
            "Acc/Forward": acc,
        }


    def train_torch_step(
        self,
        passages: BatchElement,
        reviews: BatchElement,
        config: TrainConfig,
    ):
        self.zero_grad()
        with self.autocast():
            logits_ij = self._inner_step(
                mode_w_grad=passages,
                mode_no_grad=reviews,
                config=config,
                f_backwards= self.torch_backwards,
            )

            logits_ji = self._inner_step(
                mode_w_grad=reviews,
                mode_no_grad=passages,
                config=config,
                f_backwards= self.torch_backwards,
            )

        self.average_gradients()
        self.clip_gradients()
        self.torch_step()

        with torch.no_grad():
            loss = self.model.contrastive_loss(logits_ij=logits_ij, logits_ji=logits_ji)
            acc = self.model.compute_accuracy(logits_ij=logits_ij, logits_ji=logits_ji)

        return {
            "Loss/Train": loss,
            "Acc/Forward": acc,
            "Model/logit_scale": self.model.logit_scale.sum(),
        }


