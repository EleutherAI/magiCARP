from typing import List

from carp.configs import ModelConfig
from carp.pytorch.model.architectures import *
from carp.pytorch.model.encoders import get_encoder
from carp.pytorch.training import BaseTrainer, register_trainer
from carp.util import generate_indices

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
        self.passage_encoder = encoder_class(config.model_path, config.model_arch)
        self.review_encoder = encoder_class(
            config.model_path, config.model_arch, self.passage_encoder.model
        )
        self.latent_dim = self.config.latent_dim
        self.pass_projector, self.rev_projector = self._make_projection_layers(
            self.config
        )
        self.logit_scale = nn.Parameter(
            torch.ones([], device=self.config.device)
            * torch.log(torch.tensor([1 / 0.07], device=self.config.device))
        )
        self.clamp_min = torch.log(torch.tensor([1 / 100], device=self.config.device))
        self.clamp_max = torch.log(torch.tensor([100], device=self.config.device))

    # We need a custom load function for this class since the encoder is shared
    def save(self, path: str):
        self.attempt_save(self.passage_encoder.model, path, "shared_encoder.pt")
        self.attempt_save(self.pass_projector, path, "pass_projector.pt")
        self.attempt_save(self.rev_projector, path, "rev_projector.pt")
        try:
            self.attempt_save(self.logit_scale, path, "logit_scale.pt")
        except:
            pass

    # similar to above we require a custom load and save function
    def load(self, path: str):
        self.passage_encoder.model = self.attempt_load(path, "shared_encoder.pt")
        self.review_encoder.model = self.passage_encoder.model

        self.pass_projector = self.attempt_load(path, "pass_projector.pt")
        self.rev_projector = self.attempt_load(path, "rev_projector.pt")

        self.logit_scale = self.attempt_load(path, "logit_scale.pt")

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
class CARPSharedEncoderTrainer(BaseTrainer):
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
        self.zero_grad()

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
