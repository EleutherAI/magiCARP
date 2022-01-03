import torch
import torch.nn.functional as F
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from dataclasses import dataclass
from carp.configs import CARPConfig, ModelConfig, TrainConfig
from carp.pytorch.data.mlm_pipeline import MLMBatchElement
from carp.pytorch.model.architectures import * 
from carp.pytorch.model.encoders import get_encoder
from carp.util import mbTokens, generate_indices
from typing import List



# CARP MLM differs from normal CARP since the first epoch will solely use an MLM objective to improve data efficiency. 
# TODO: The learning rate scheduler needs to account for this, so we need a way to register custom LR schedulers.
# TODO: We need to make sure it saves a CARP MLM checkpoint after the first epoch so that we can convert it to CARP Cloob or CARP momentum

patch_typeguard()

@typechecked
@register_architecture
class CARPMLM(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        encoder_class = get_encoder(config.encoder_type)
        self.passage_encoder = encoder_class(
            config.model_path, config.model_arch
        )
        self.review_encoder = encoder_class(
            config.model_path, config.model_arch
        )
        self.latent_dim = config.latent_dim
        self.pass_projector, self.rev_projector = self._make_projection_layers(config)
        self.logit_scale = nn.Parameter(
            torch.ones([], device=config.device)
            * torch.log(torch.tensor([1 / 0.07], device=config.device))
        )
        self.clamp_min = torch.log(torch.tensor([1 / 100], device=config.device))
        self.clamp_max = torch.log(torch.tensor([100], device=config.device))

        self.mlm_mode = True

    def _embed_data(
        self,
        x: MLMBatchElement,
        encoder,
        projector,
    ):
        if self.mlm_mode:
            x = encoder(x.input_ids.to(self.device),
                x.mask.to(self.device),
                x.mlm_input_ids.to(self.device),
                x.mlm_labels.to(self.device))
        else:
            x = encoder(x.input_ids.to(self.device),
                x.mask.to(self.device),
                None)
            # if we are not in mlm mode, run the projection layer which is used for contrastive learning 
            if not self.mlm_mode:
                return projector(x)
        return x
    # overridden to decrease memory footprint
    def calculate_embeddings(
        self,
        passages: Iterable[
            Tuple[
                BatchElement
            ]
        ],
        reviews: Iterable[
            Tuple[
                BatchElement
            ]
        ],
        return_only_embeddings : bool = True,
    ):
        # Get encodings without grad
        with torch.no_grad(), torch.cuda.amp.autocast():
            pass_encs = [self.encode_passages(p) for p in passages]
        with torch.no_grad(), torch.cuda.amp.autocast():
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
        scaler: torch.cuda.amp.GradScaler
    ) -> Dict[str, TensorType[()]]:


        
        microbatch_inds = generate_indices(
            passages.input_ids.shape[0], config.microbatch_size, shuffle=False
        )
        # Split batch elements into smaller batch elements 
        pass_mbs: List[Tuple[MLMBatchElement]] = [
            MLMBatchElement(passages.input_ids[i], passages.mask[i],
                passages.mlm_input_ids[i], passages.mlm_labels[i])\
                for i in microbatch_inds
        ]
        rev_mbs: List[Tuple[MLMBatchElement]] = [
            MLMBatchElement(reviews.input_ids[i], reviews.mask[i],
                reviews.mlm_input_ids[i], reviews.mlm_labels[i])\
                 for i in microbatch_inds
        ]
        self.zero_grad(opt)

        if self.mlm_mode:
            with torch.cuda.amp.autocast():
                loss_pass = [self.encode_passages(p).loss for p in pass_mbs]
            [scaler.scale(l).backward() for l in loss_pass]
            with torch.cuda.amp.autocast():
                loss_rev = [self.encode_reviews(r).loss for r in rev_mbs]
            [scaler.scale(l).backward() for l in loss_rev]
            self.step(scaler, opt)

            mlm_loss = (sum(loss_pass) + sum(loss_rev)) / (passages.input_ids.shape[0] // config.microbatch_size)
            return {
                "Loss/MLM" : mlm_loss,
                "Loss/Train" : mlm_loss
            }
                
        # Initially get all encodings without grad
        pass_encs, rev_encs = self.calculate_embeddings(pass_mbs, rev_mbs)

        # Encode passages in microbatches (with grad)
        for index, passage in enumerate(pass_mbs):
            passage, mask = passage
            pass_tmp = pass_encs.copy()
            with torch.cuda.amp.autocast():
                pass_tmp[index] = self.encode_passages(
                    passage.to(self.device), mask.to(self.device)
                )
                loss, forward_acc = self.contrastive_loss(
                    torch.cat(pass_tmp), torch.cat(rev_encs)
                )
            scaler.scale(loss).backward()
        # Encode reviews in microbatches (with grad)
        for index, review in enumerate(rev_mbs):
            review, mask = review
            rev_tmp = rev_encs.copy()  # no_grad
            with torch.cuda.amp.autocast():
                rev_tmp[index] = self.encode_reviews(
                    review.to(self.device), mask.to(self.device)
                )  # grad _just_ at positions in `index`
                loss, _ = self.contrastive_loss(
                    torch.cat(pass_encs), torch.cat(rev_tmp)
                )
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
