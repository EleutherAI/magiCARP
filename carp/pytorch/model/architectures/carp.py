from typing import List

from carp.configs import ModelConfig
from carp.pytorch.model.architectures import *
from carp.pytorch.training import BaseTrainer, register_trainer
from carp.pytorch.scalability_utils import print_rank_0
from carp.util import generate_indices

patch_typeguard()


@typechecked
@register_architecture
class CARP(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)

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
class CARPTrainer(BaseTrainer):
    def train_deepspeed_step(
        self,
        passages: BatchElement,
        reviews: BatchElement,
        config: TrainConfig,
    ):
        forward_output = self.model(passages, reviews, config)

        # Encode passages in microbatches (with grad)
        for index, passage in enumerate(forward_output["pass_mbs"]):
            pass_tmp = forward_output["pass_encs"].copy()
            with torch.cuda.amp.autocast():
                pass_tmp[index] = self.model.module.encode_passages(passage).hidden
            # torch.float32 torch.float16
            loss = self.model.module.contrastive_loss(
                torch.cat(pass_tmp), torch.cat(forward_output["rev_encs"])
            )
            self.model.backward(loss)

        # Encode reviews in microbatches (with grad)
        for index, review in enumerate(forward_output["rev_mbs"]):
            rev_tmp = forward_output["rev_encs"].copy()  # no_grad
            with torch.cuda.amp.autocast():
                rev_tmp[index] = self.model.module.encode_reviews(review).hidden
            # grad _just_ at positions in `index`
            loss = self.model.module.contrastive_loss(
                torch.cat(forward_output["pass_encs"]), torch.cat(rev_tmp)
            )
            self.model.backward(loss)

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
        self.zero_grad(self.opt)

        # Encode passages in microbatches (with grad)
        for index, passage in enumerate(forward_output["pass_mbs"]):
            pass_tmp = forward_output["pass_encs"].copy()
            with torch.cuda.amp.autocast():
                pass_tmp[index] = self.model.encode_passages(passage).hidden
                loss = self.model.contrastive_loss(
                    torch.cat(pass_tmp), torch.cat(forward_output["rev_encs"])
                )
            self.scaler.scale(loss).backward()
        # Encode reviews in microbatches (with grad)
        for index, review in enumerate(forward_output["rev_mbs"]):
            rev_tmp = forward_output["rev_encs"].copy()  # no_grad
            with torch.cuda.amp.autocast():
                rev_tmp[index] = self.model.encode_reviews(review).hidden
                # grad _just_ at positions in `index`
                loss = self.model.contrastive_loss(
                    torch.cat(forward_output["pass_encs"]), torch.cat(rev_tmp)
                )
            self.scaler.scale(loss).backward()
        # Clipping
        if self.model.config.grad_clip != -1:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.model.config.grad_clip
            )

        self.torch_step()

        return {
            "Loss/Train": loss,
            "Acc/Forward": forward_output["forward_acc"],
        }
