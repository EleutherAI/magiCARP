from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

from carp.configs import ModelConfig
from carp.pytorch.data.mlm_pipeline import MLMBatchElement
from carp.pytorch.model.architectures import *
from carp.pytorch.training.trainer import BaseTrainer, register_trainer
from carp.util import generate_indices, get_scheduling_func

# CARP MLM differs from normal CARP since the first epoch will solely use an MLM objective to improve data efficiency.
# TODO: The learning rate scheduler needs to account for this, so we need a way to register custom LR schedulers.
# TODO: We need to make sure it saves a CARP MLM checkpoint after the first epoch so that we can convert it to CARP Cloob or CARP momentum
# TODO: DEEPSPEED SUPPORT (kevin)

patch_typeguard()


@typechecked
@register_architecture
class CARPMLM(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.mlm_mode = True

    def _embed_data(
        self,
        x: MLMBatchElement,
        encoder,
        projector,
    ):
        if self.mlm_mode:
            x = encoder(
                x.input_ids.to(self.device),
                x.mask.to(self.device),
                x.mlm_input_ids.to(self.device),
                x.mlm_labels.to(self.device),
            )
        else:
            x = encoder(x.input_ids.to(self.device), x.mask.to(self.device), None)
            # if we are not in mlm mode, run the projection layer which is used for contrastive learning
            x.hidden = projector(x.hidden)
            return x
        return x

    # overridden to decrease memory footprint
    def calculate_embeddings(
        self,
        passages: Iterable[Tuple[BatchElement]],
        reviews: Iterable[Tuple[BatchElement]],
        return_only_embeddings: bool = True,
    ):
        # Get encodings without grad
        with no_grad(),self.autocast():
            pass_encs = [self.encode_passages(p) for p in passages]
        with no_grad(),self.autocast():
            rev_encs = [self.encode_reviews(r) for r in reviews]

        # if we only need the embeddings, fetch them
        if return_only_embeddings:
            pass_encs = list(map(lambda x: x.hidden, pass_encs))
            rev_encs = list(map(lambda x: x.hidden, rev_encs))
        return pass_encs, rev_encs

    def train_step(
        self,
        passages: MLMBatchElement,
        reviews: MLMBatchElement,
        config: TrainConfig,
        opt: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
    ) -> Dict[str, TensorType[()]]:

        microbatch_inds = generate_indices(
            passages.input_ids.shape[0], config.microbatch_size, shuffle=False
        )
        # Split batch elements into smaller batch elements
        pass_mbs: List[MLMBatchElement] = [
            MLMBatchElement(
                passages.input_ids[i],
                passages.mask[i],
                passages.mlm_input_ids[i],
                passages.mlm_labels[i],
            )
            for i in microbatch_inds
        ]
        rev_mbs: List[MLMBatchElement] = [
            MLMBatchElement(
                reviews.input_ids[i],
                reviews.mask[i],
                reviews.mlm_input_ids[i],
                reviews.mlm_labels[i],
            )
            for i in microbatch_inds
        ]
        self.zero_grad()

        def accum_every_mb(inp: MLMBatchElement, passage: bool = True):
            with self.autocast():
                if passage:
                    loss = self.encode_passages(inp).loss
                else:
                    loss = self.encode_reviews(inp).loss
            scaler.scale(loss).backward()
            return loss.detach()

        if self.mlm_mode:
            loss_pass = [accum_every_mb(p) for p in pass_mbs]
            loss_rev = [accum_every_mb(r, False) for r in rev_mbs]
            self.step(scaler, opt)

            mlm_loss = (sum(loss_pass) + sum(loss_rev)) / (
                passages.input_ids.shape[0] // config.microbatch_size
            )
            return {"Loss/MLM": mlm_loss, "Loss/Train": mlm_loss}

        # Initially get all encodings without grad
        pass_encs, rev_encs = self.calculate_embeddings(pass_mbs, rev_mbs)

        # compute accuracy
        forward_acc = self.compute_accuracy(torch.cat(pass_encs), torch.cat(rev_encs))

        # Encode passages in microbatches (with grad)
        for index, passage in enumerate(pass_mbs):
            pass_tmp = pass_encs.copy()
            with self.autocast():
                pass_tmp[index] = self.encode_passages(passage).hidden
                loss = self.contrastive_loss(torch.cat(pass_tmp), torch.cat(rev_encs))
            scaler.scale(loss).backward()
        # Encode reviews in microbatches (with grad)
        for index, review in enumerate(rev_mbs):
            rev_tmp = rev_encs.copy()  # no_grad
            with self.autocast():
                rev_tmp[index] = self.encode_reviews(review).hidden
                # grad _just_ at positions in `index`
                loss = self.contrastive_loss(torch.cat(pass_encs), torch.cat(rev_tmp))
            scaler.scale(loss).backward()

        # Clipping
        if self.config.grad_clip != -1:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(self.parameters(), config.grad_clip)

        self.step(scaler, opt)
        return {
            "Loss/Contrastive": loss,
            "Loss/Train": loss,
            "Acc/Forward": forward_acc,
        }


@register_trainer
class CARPMLMTrainer(BaseTrainer):
    def __init__(self, config: TrainConfig):
        self.epoch_number = 0
        self.mlm_mode = False
        super().__init__(config)

    def on_epoch_start(
        self, model: BaseModel, scheduler: _LRScheduler, opt: Optimizer, **kwargs
    ) -> Tuple[BaseModel, _LRScheduler, Optimizer]:

        if self.epoch_number == 0:
            self.mlm_mode = True
        if self.epoch_number == 1:
            self.mlm_mode = False
            # change to the batch size required for the contrastive learning component
            self.train_config.batch_size = 2048
            # reset the LR scheduler
            scheduler = LambdaLR(opt, get_scheduling_func(self.train_config))
            # change the gradient accum steps to 1
            model.config.grad_accum = 1

        self.epoch_number += 1
        return model, scheduler, opt

    # TODO: Add training steps and deepspeed support!
