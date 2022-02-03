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


# Uses only a single encoder for both the passage encoder and critique encoder. Prepends modality specific tokens
# per encoder

patch_typeguard()

@typechecked
@register_architecture
class CARPSharedEncoder(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config, skip_init=True)
        self.config = config
        encoder_class = get_encoder(config.encoder_type)
        self.passage_encoder = encoder_class(
            config.model_path, config.model_arch
        )
        self.review_encoder = encoder_class(
            config.model_path, config.model_arch, self.passage_encoder.model
        )
        self.latent_dim = self.config.latent_dim
        self.pass_projector, self.rev_projector = self._make_projection_layers(self.config)
        self.logit_scale = nn.Parameter(
            torch.ones([], device=self.config.device)
            * torch.log(torch.tensor([1 / 0.07], device=self.config.device))
        )
        self.clamp_min = torch.log(torch.tensor([1 / 100], device=self.config.device))
        self.clamp_max = torch.log(torch.tensor([100], device=self.config.device))
        
    
    # We need a custom load function for this class since the encoder is shared 
    def save(self, path : str):
        self.attempt_save(self.passage_encoder.model, path, "shared_encoder.pt")
        self.attempt_save(self.pass_projector, path, "pass_projector.pt")
        self.attempt_save(self.rev_projector, path, "rev_projector.pt")
        try:
            self.attempt_save(self.logit_scale, path, "logit_scale.pt")
        except:
            pass

    # similar to above we require a custom load and save function
    def load(self, path : str):
        self.passage_encoder.model = self.attempt_load(path, "shared_encoder.pt")
        self.review_encoder.model = self.passage_encoder.model

        self.pass_projector = self.attempt_load(path, "pass_projector.pt")
        self.rev_projector = self.attempt_load(path, "rev_projector.pt")

        self.logit_scale = self.attempt_load(path, "logit_scale.pt")
        
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)

        self.step(scaler, opt)
        return {
            "Loss/Train": loss,
            "Acc/Forward": forward_acc,
        }
