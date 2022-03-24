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
    def item_divergence__mode_i_to_mode_j(
        self,
        x: TensorType[-1, "latent_dim"],
        y: TensorType[-1, "latent_dim"],
        normalize=False,
    ):
        """
        Returns a score comparing item x to item y. For most purposes this
        will be a similarity or distance metric, i.e. divergence(x,y) = divergence(y,x).
        """
        return self.cosine_sim(x, y, normalize)

    def item_divergence__mode_j_to_mode_i(
        self,
        x, #: TensorType[-1, "latent_dim"],
        y, #: TensorType[-1, "latent_dim"],
        normalize=False,
    ):
        """
        Returns a score comparing item x to item y. For most purposes this
        will be a similarity or distance metric, i.e. divergence(x,y) = divergence(y,x).
        Defaults behavior assumes a symmetric divergence measure. If this is not
        the case, be sure to override this function. If this function is not overridden,
        item_divergence__mode_i_to_mode_j(y,x) will be used to compute the j->i divergence.
        """
        return self.item_divergence__mode_i_to_mode_j(x=y,y=x, normalize=normalize)

    def item_logits__mode_i_to_mode_j(
        self,
        x: TensorType[-1, "latent_dim"],
        y: TensorType[-1, "latent_dim"],
        normalize=False,
    ) -> TensorType:
        S_ij = self.item_divergence__mode_i_to_mode_j(x=x,y=y,normalize=normalize)
        return S_ij * self.logit_scale.exp()
    
    def item_logits__mode_j_to_mode_i(
        self,
        x, #: TensorType[-1, "latent_dim"],
        y, #: TensorType[-1, "latent_dim"],
        normalize=False,
        ) -> TensorType:
        S_ji = self.item_divergence__mode_j_to_mode_i(x=x,y=y,normalize=normalize)
        return S_ji * self.logit_scale.exp()

    def _compute_loss_or_acc(
        self, 
        x=None, #: TensorType[-1, "latent_dim"], 
        y=None, #: TensorType[-1, "latent_dim"],
        normalize=False,
        logits_ij = None,
        return_loss = True,
        return_acc=True,
    ):
        if isinstance(logits_ij, list):
            logits_ij = torch.cat(logits_ij)
        if x is not None:
            n = x.shape[0]
        else:
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

    def logits_ij_to_loss_ij(self, logits_ij):
        #logger.debug(logits_ij.shape)
        n = logits_ij.shape[-1]
        labels = torch.arange(n, device=self.config.device)
        loss_ij = F.cross_entropy(logits_ij, labels)
        #logger.debug(loss_ij.shape) # has no shape, scalar. 
        return loss_ij



    def loss_component__mode_i_to_mode_j(
        self, 
        x, #: TensorType[-1, "latent_dim"], 
        y, #: TensorType[-1, "latent_dim"],
        normalize=False,
        logits_ij = None,
    ) -> TensorType[(), float]:
        assert not all (obj is None for obj in (x,y,logits_ij))

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
        assert not all (obj is None for obj in (x,y,logits_ji))
        return self.loss_component__mode_i_to_mode_j(x=y,y=x,normalize=normalize, logits_ij=logits_ji)


    #def contrastive_loss_terms(
    #    self, 
    #    x, #: TensorType[-1, "latent_dim"], 
    #    y, #: TensorType[-1, "latent_dim"],
    #    normalize=False,
    #    logits_ij=None,
    #    logits_ji=None,
    #) -> TensorType[(), float]:
    #
    #    loss_ij = self.loss_component__mode_i_to_mode_j(x,y,normalize,logits_ij = logits_ij)
    #    loss_ji = self.loss_component__mode_j_to_mode_i(x,y,normalize,logits_ij = logits_ji)
    #    return loss_ij, loss_ji

    def loss_components(
        self, 
        x=None,
        y=None,
        normalize=False,
        loss_ij=None,
        loss_ji=None,
        #use_loss_transpose=True,
        use_loss_transpose=False,
        logits_ij=None,
        logits_ji=None,
    ):
        assert not all (obj is None for obj in (x,y,logits_ij, logits_ji))
        if loss_ij is None:
            assert not all (obj is None for obj in (x,y))
            loss_ij = self.loss_component__mode_i_to_mode_j(x=x,y=y,normalize=normalize, logits_ij=logits_ij)
        if loss_ji is None:
            loss_ji = loss_ij.T
            if not use_loss_transpose:
                assert not all (obj is None for obj in (x,y))
                loss_ji = self.loss_component__mode_j_to_mode_i(x=x,y=y,normalize=normalize, logits_ij=logits_ji)
        return loss_ij, loss_ji

    # def contrastive_loss(
    #     self, 
    #     x=None,
    #     y=None,
    #     normalize=False,
    #     loss_ij=None,
    #     loss_ji=None,
    #     #use_loss_transpose=True,
    #     use_loss_transpose=False,
    #     logits_ij=None,
    #     logits_ji=None,
    # ) -> TensorType[(), float]:
    #     if loss_ij is not None and loss_ji is not None:
    #         losses = loss_ij, loss_ji
    #     losses = self.loss_components(
    #         x=x,y=y, normalize=normalize, 
    #         loss_ij=loss_ij, loss_ji=loss_ji, use_loss_transpose=use_loss_transpose,
    #         logits_ij=logits_ij,
    #         logits_ji=logits_ji
    #         )
        
    #     return sum(losses) / len(losses)

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


@typechecked
@register_architecture
@typechecked
@register_architecture
class CARPFilip(CARPSimRefactor):

    def item_divergence__mode_i_to_mode_j(
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
        S_ij = self.item_divergence__mode_i_to_mode_j(x,y,normalize)
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
        logit_chunks_ij=None,
        logit_chunks_ji=None,
        #logits_ji=None,
        f_backwards=None,
    ):
        # this... doesn't go here.
        batch_size = config.batch_size
        if not self.model.training:
            batch_size = config.validation_size

        microbatch_inds = generate_indices(
            #config.batch_size, #mode_w_grad.input_ids.shape[0],
            batch_size,
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
        if logit_chunks_ij is None:
            logit_chunks_ij = []
            compute_loss = False
        for i, item in enumerate(mbs_w_grad):
            with torch.cuda.amp.autocast():
                enc_i = encoder_w_grad(item).hidden

            logits_ij_list = [logits_func(enc_i, enc_no_grad) 
                for enc_no_grad in mbs_enc_no_grad]
            logits_ij = torch.cat(logits_ij_list, dim=-1) # [microbatch_size batch_size]

            if compute_loss:
                assert logit_chunks_ji is not None
                temp_logits = logit_chunks_ij.copy()
                temp_logits[i] = logits_ij
                logits_cat = torch.cat(temp_logits) # logits_cat.shape -> [batch_size batch_size]
                logits_ji = torch.cat(logit_chunks_ji.copy())
                loss_ij = self.model.logits_ij_to_loss_ij(logits_cat)
                loss_ji = self.model.logits_ij_to_loss_ij(logits_ji)
                loss = (loss_ij + loss_ji)/2
                f_backwards(loss)
            else:
                logit_chunks_ij.append(logits_ij)
        return logit_chunks_ij
    
    def microbatch_up_logits__mode_j_to_mode_i(
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

        return self.microbatch_up_logits__mode_i_to_mode_j(
            mode_w_grad=mode_no_grad,
            mode_no_grad=mode_w_grad,
            encoder_w_grad=encoder_no_grad,
            encoder_no_grad=encoder_w_grad,
            logits_func=logits_func,
            config=config,
            logit_chunks=logit_chunks,
            f_backwards=f_backwards,
        )

    def compute_logits_no_grad(self,
        mode_w_grad: BatchElement,
        mode_no_grad: BatchElement,
        config: TrainConfig,
        f_backwards=None,
    ):

        # 1. Build up logits no grad
        logger.debug("building logits_ij no grad")
        with torch.no_grad():
            logits_chunks_ij = self.microbatch_up_logits__mode_i_to_mode_j(
                mode_w_grad=mode_w_grad,
                mode_no_grad=mode_no_grad,
                encoder_w_grad=self.model.encode_passages,
                encoder_no_grad=self.model.encode_reviews,
                logits_func=self.model.item_logits__mode_i_to_mode_j,
                config=config,
            )

        logger.debug("building logits_ji no grad")
        with torch.no_grad():
            logits_chunks_ji = self.microbatch_up_logits__mode_i_to_mode_j(
                mode_w_grad=mode_no_grad,
                mode_no_grad=mode_w_grad,
                encoder_w_grad=self.model.encode_reviews,
                encoder_no_grad=self.model.encode_passages,
                logits_func=self.model.item_logits__mode_i_to_mode_j,
                config=config,
            )
        return logits_chunks_ij, logits_chunks_ji

    def _inner_step(self,
        mode_w_grad: BatchElement,
        mode_no_grad: BatchElement,
        config: TrainConfig,
        f_backwards=None,
    ):

        logits_chunks_ij, logits_chunks_ji = self.compute_logits_no_grad(
            mode_w_grad=mode_w_grad,
            mode_no_grad=mode_no_grad,
            config=config,
        )
        
        # 2. Ok, again but this time with grad and loss....
        logger.debug("building logits_ij with grad")
        logits_chunks_ij = self.microbatch_up_logits__mode_i_to_mode_j(
                mode_w_grad=mode_w_grad,
                mode_no_grad=mode_no_grad,
                encoder_w_grad=self.model.encode_passages,
                encoder_no_grad=self.model.encode_reviews,
                logits_func=self.model.item_logits__mode_i_to_mode_j,
                config=config,
                logit_chunks_ij=logits_chunks_ij,
                logit_chunks_ji=logits_chunks_ji,
                f_backwards=f_backwards,
            )
        
        logger.debug("building logits_ji with grad")
        logits_chunks_ij = self.microbatch_up_logits__mode_i_to_mode_j(
                mode_w_grad=mode_no_grad,
                mode_no_grad=mode_w_grad,
                encoder_w_grad=self.model.encode_reviews,
                encoder_no_grad=self.model.encode_passages,
                logits_func=self.model.item_logits__mode_i_to_mode_j,
                config=config,
                logit_chunks_ij=logits_chunks_ji,
                logit_chunks_ji=logits_chunks_ij, # with s or without s??? bad david, bad. pick one.
                f_backwards=f_backwards,
            )

        return torch.cat(logits_chunks_ij), torch.cat(logits_chunks_ji)
    
    def train_deepspeed_step(
        self,
        passages: BatchElement,
        reviews: BatchElement,
        config: TrainConfig,
    ):
        with self.autocast():
            logits_ij, logits_ji = self._inner_step(
                mode_w_grad=passages,
                mode_no_grad=reviews,
                config=config,
                f_backwards= self.deepspeed_backwards,
            )
        
        self.average_gradients()
        self.clip_gradients()
        self.deepspeed_step()

        with torch.no_grad():
            loss = (logits_ij + logits_ji)/2
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
            logits_ij, logits_ji = self._inner_step(
                mode_w_grad=passages,
                mode_no_grad=reviews,
                config=config,
                f_backwards= self.torch_backwards,
            )
        
        self.average_gradients()
        self.clip_gradients()
        self.torch_step()

        with torch.no_grad():
            loss_ij = self.model.logits_ij_to_loss_ij(logits_ij)
            loss_ji = self.model.logits_ij_to_loss_ij(logits_ji)
            loss = (loss_ij + loss_ji)/2
            acc = self.model.compute_accuracy(logits_ij=logits_ij, logits_ji=logits_ji)

        return {
            "Loss/Train": loss,
            "Acc/Forward": acc,
            "Model/logit_scale": self.model.logit_scale.sum(),
        }

    def eval_step(self, dataset):
        """
        Runs a single evaluation step on the model.
        Args:
            dataset: the validation dataset
        Returns:
            dict: Dictionary of validation loss and validation accuracy
        """
        loss = 0
        acc = 0
        n=0
        for passages, reviews in dataset:
            with torch.no_grad():
                logits_chunks_ij, logits_chunks_ji = self.compute_logits_no_grad(
                    mode_w_grad=passages,#: BatchElement,
                    mode_no_grad=reviews,#: BatchElement,
                    #config=config,#: TrainConfig,
                    # uh... why not just do this?
                    config=self.train_config,
                )
                logits_ij, logits_ji = torch.cat(logits_chunks_ij), torch.cat(logits_chunks_ji)
                loss_ij = self.model.logits_ij_to_loss_ij(logits_ij)
                loss_ji = self.model.logits_ij_to_loss_ij(logits_ji)
                loss_k = (loss_ij + loss_ji)/2
                acc_k = self.model.compute_accuracy(logits_ij=logits_ij, logits_ji=logits_ji)
                loss += loss_k.item()
                acc += acc_k.item()
                n+=1
        return {"Loss/Validation": loss/n, "Acc/Validation": acc/n}

    def eval_step__OLD(self, dataset):
        """
        Runs a single evaluation step on the model.
        Args:
            dataset: the validation dataset
        Returns:
            dict: Dictionary of validation loss and validation accuracy
        """
        passages = []
        reviews = []
        for p, r in dataset:
            passages.append(p)
            reviews.append(r)

        # TODO: Ideally should get microbatch size from trainconfig for the second argument
        # DMARX: why not just replace "8" with self.train_config.microbatch_size then?
        passages = chunkBatchElement(passages[0], 8)
        reviews = chunkBatchElement(reviews[0], 8)

        with no_grad():
            if self.use_deepspeed:
                pass_emb, rev_emb = self.model.module.calculate_embeddings(
                    passages, reviews
                )
                val_loss = self.model.module.contrastive_loss(
                    torch.cat(pass_emb), torch.cat(rev_emb)
                )
                val_acc = self.model.module.compute_accuracy(
                    torch.cat(pass_emb), torch.cat(rev_emb)
                )
            else:
                pass_emb, rev_emb = self.model.calculate_embeddings(passages, reviews)
                val_loss = self.model.contrastive_loss(
                    torch.cat(pass_emb), torch.cat(rev_emb)
                )
                val_acc = self.model.compute_accuracy(
                    torch.cat(pass_emb), torch.cat(rev_emb)
                )

        return {"Loss/Validation": val_loss.item(), "Acc/Validation": val_acc.item()}

