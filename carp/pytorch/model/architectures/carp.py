from typing import List

from carp.configs import ModelConfig
from carp.pytorch.model.architectures import *
from carp.pytorch.scalability_utils import print_rank_0
from carp.pytorch.training.trainer import BaseTrainer, register_trainer
from carp.util import generate_indices
import torch.distributed as dist

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
        with self.autocast():
            forward_output = self.model(passages, reviews, config)

        rev_encs, all_rev_encs, rev_offset = self.contrastive_parallel_all_gather(
            forward_output["rev_encs"]
        )
        pass_encs, all_pass_encs, pass_offset = self.contrastive_parallel_all_gather(
            forward_output["pass_encs"]
        )

        # Encode passages in microbatches (with grad)
        for index, passage in enumerate(forward_output["pass_mbs"]):
            pass_tmp = list(all_pass_encs.clone().split(config.microbatch_size))
            with self.autocast():
                micro_batch = self.model.module.encode_passages(passage).hidden
                pass_tmp[pass_offset + index] = micro_batch
                loss = self.model.module.contrastive_loss(
                    torch.cat(pass_tmp), all_rev_encs
                )
            self.deepspeed_backwards(loss)

        # Encode reviews in microbatches (with grad)
        for index, review in enumerate(forward_output["rev_mbs"]):
            rev_tmp = list(all_rev_encs.clone().split(config.microbatch_size))
            with self.autocast():
                micro_batch = self.model.module.encode_reviews(review).hidden
                rev_tmp[rev_offset + index] = micro_batch
                loss = self.model.module.contrastive_loss(
                    all_pass_encs, torch.cat(rev_tmp)
                )
            self.deepspeed_backwards(loss)

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
        with self.autocast():
            forward_output = self.model(passages, reviews, config)

        # does gradient accumulation
        self.zero_grad()

        # Encode passages in microbatches (with grad)
        for index, passage in enumerate(forward_output["pass_mbs"]):
            pass_tmp = forward_output["pass_encs"].copy()
            with self.autocast():
                pass_tmp[index] = self.model.encode_passages(passage).hidden
                loss = self.model.contrastive_loss(
                    torch.cat(pass_tmp), torch.cat(forward_output["rev_encs"])
                )

            self.torch_backwards(loss)

        # Encode reviews in microbatches (with grad)
        for index, review in enumerate(forward_output["rev_mbs"]):
            rev_tmp = forward_output["rev_encs"].copy()  # no_grad
            with self.autocast():
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
            "Model/logit_scale": self.model.logit_scale.sum(),
        }
