import torch
import torch.nn.functional as F
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from carp.configs import CARPConfig, ModelConfig, TrainConfig
from carp.pytorch.model.architectures import * 
from carp.pytorch.model.encoders import get_encoder
from carp.util import mbTokens, generate_indices
from typing import List

from carp.pytorch.data.utils.data_util import BatchElement

patch_typeguard()

@typechecked
@register_architecture
class CARP(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        

    def train_step(
        self,
        passages: BatchElement,
        reviews: BatchElement,
        config: TrainConfig,
        opt: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
    ) -> Dict[str, TensorType[()]]:
        microbatch_inds = generate_indices(
            passages.input_ids.shape[0], config.microbatch_size, shuffle=False
        )
        # Split tokens and masks into these microbatches
        pass_mbs: List[BatchElement] = [
            BatchElement(passages.input_ids[i], passages.mask[i]) for i in microbatch_inds
        ]
        rev_mbs: List[BatchElement] = [
            BatchElement(reviews.input_ids[i], reviews.mask[i]) for i in microbatch_inds
        ]

        # Initially get all encodings without grad
        pass_encs, rev_encs = self.calculate_embeddings(pass_mbs, rev_mbs)

        #compute accuracy
        forward_acc = self.compute_accuracy(torch.cat(pass_encs), torch.cat(rev_encs))

        # does gradient accumulation
        self.zero_grad(opt)

        # Encode passages in microbatches (with grad)
        for index, passage in enumerate(pass_mbs):
            pass_tmp = pass_encs.copy()
            with torch.cuda.amp.autocast():
                pass_tmp[index] = self.encode_passages(passage).hidden
                loss  = self.contrastive_loss(
                    torch.cat(pass_tmp), torch.cat(rev_encs)
                )
            scaler.scale(loss).backward()
        # Encode reviews in microbatches (with grad)
        for index, review in enumerate(rev_mbs):
            rev_tmp = rev_encs.copy()  # no_grad
            with torch.cuda.amp.autocast():
                rev_tmp[index] = self.encode_reviews(review).hidden  
                # grad _just_ at positions in `index`
                loss = self.contrastive_loss(
                    torch.cat(pass_encs), torch.cat(rev_tmp)
                )
            scaler.scale(loss).backward()
        # Clipping
        if self.config.grad_clip != -1:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(self.parameters(), config.grad_clip)

        self.step(scaler, opt)
        return {
            "Loss/Train": loss,
            "Acc/Forward": forward_acc,
        }
