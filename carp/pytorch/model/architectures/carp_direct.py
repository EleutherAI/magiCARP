from typing import List

from carp.configs import ModelConfig
from carp.pytorch.model.architectures import *
from carp.pytorch.scalability_utils import print_rank_0
from carp.pytorch.training.trainer import BaseTrainer, register_trainer
from carp.util import generate_indices
import torch.distributed as dist

from torch import nn

patch_typeguard()

@typechecked
@register_architecture
class CARPDirect(BaseModel):

    # Slightly modified init to replace projectors
    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # CARPDirect requires latent dim is less than encoders d_model
        assert self.latent_dim <= self.passage_encoder.d_model and \
            self.latent_dim <= self.review_encoder.d_model
        
        # Rather than projecting embedding to higher dim latent space
        # Take slice of embedding as latent vector
        class GenericProjector(nn.Module):
            def __init__(self, latent_dim):
                super().__init__()
                self.latent_dim = latent_dim
            
            def forward(self, x):
                return x[:,:self.latent_dim]
        
        self.pass_projector = GenericProjector(self.latent_dim)
        self.rev_projector = GenericProjector(self.latent_dim)

    # Identical to CARP.forward
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
    
    # Override base save just to skip projectors
    def save(self, path: str):
        self.attempt_save(self.passage_encoder.model, path, "passage_encoder.pt")
        self.attempt_save(self.review_encoder.model, path, "review_encoder.pt")

        #self.attempt_save(self.pass_projector, path, "pass_projector.pt")
        #self.attempt_save(self.rev_projector, path, "rev_projector.pt")
        try:
            self.attempt_save(self.logit_scale, path, "logit_scale.pt")
        except:
            pass

    # Override base load just to skip projectors
    def load(self, path: str):
        self.passage_encoder.model = self.attempt_load(path, "passage_encoder.pt")
        self.review_encoder.model = self.attempt_load(path, "review_encoder.pt")

        #self.pass_projector = self.attempt_load(path, "pass_projector.pt")
        #self.rev_projector = self.attempt_load(path, "rev_projector.pt")

        self.logit_scale = self.attempt_load(path, "logit_scale.pt")


