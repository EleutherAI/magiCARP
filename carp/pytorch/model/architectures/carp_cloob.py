from typing import List

from carp.configs import ModelConfig
from carp.pytorch.model.architectures import *
from carp.pytorch.training.trainer import BaseTrainer, register_trainer
from carp.util import generate_indices


# TODO: Add torch typing
# yoinked from https://github.com/ml-jku/cloob
def infoLOOB_loss(x, y, labels, logit_scale):
    exp_logit_scale = logit_scale.exp()
    logits = x @ y.T * exp_logit_scale

    positives = -torch.mean(torch.sum(logits * labels, dim=1))

    # For logsumexp the zero entries must be equal to a very large negative number
    large_neg = -1000.0
    arg_lse = logits * torch.logical_not(labels) + labels * large_neg
    negatives = torch.mean(torch.logsumexp(arg_lse, dim=1))
    return (1 / exp_logit_scale) * (positives + negatives)


def hopfield_retrieval(image_features, text_features, hopfield_scale):
    patterns_xx = hopfield(
        state_patterns=image_features,
        stored_patterns=image_features,
        hopfield_scale=hopfield_scale,
    )
    patterns_yy = hopfield(
        state_patterns=text_features,
        stored_patterns=text_features,
        hopfield_scale=hopfield_scale,
    )
    patterns_xy = hopfield(
        state_patterns=text_features,
        stored_patterns=image_features,
        hopfield_scale=hopfield_scale,
    )
    patterns_yx = hopfield(
        state_patterns=image_features,
        stored_patterns=text_features,
        hopfield_scale=hopfield_scale,
    )

    return patterns_xx, patterns_yy, patterns_xy, patterns_yx


def hopfield(state_patterns, stored_patterns, hopfield_scale):
    retrieved_patterns = stored_patterns.T @ nn.functional.softmax(
        hopfield_scale.exp() * stored_patterns @ state_patterns.t(), dim=0
    )
    # Column vectors -> dim=0 to normalize the column vectors
    retrieved_patterns = retrieved_patterns / retrieved_patterns.norm(
        dim=0, keepdim=True
    )
    return retrieved_patterns


patch_typeguard()


@typechecked
@register_architecture
class CARPCloob(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config, skip_init=True)

        # Run the normal CARP init since we are skipping it
        self.config = config
        encoder_class = get_encoder(config.encoder_type)
        self.passage_encoder = encoder_class(config.model_path, config.model_arch, config.tokenizer_path)
        self.review_encoder = encoder_class(config.model_path, config.model_arch, config.tokenizer_path)

        self.latent_dim = self.config.latent_dim
        self.pass_projector, self.rev_projector = self._make_projection_layers(
            self.config
        )

        self.clamp_min = torch.log(
            torch.tensor([1 / 100], device=self.config.device)
        )
        self.clamp_max = torch.log(torch.tensor([100], device=self.config.device))

        # Add cloob specific parameters
        self.hopfield_scale = torch.ones([], device=self.config.device) * torch.log(
            torch.tensor([8], device=self.config.device, requires_grad=False)
        )
        self.logit_scale = torch.ones([], device=self.config.device) *\
             torch.log(torch.tensor([30], device=self.config.device, requires_grad=False))

        self.clamp_min = torch.log(torch.tensor([1 / 100], device=self.config.device))
        self.clamp_max = torch.log(torch.tensor([100], device=self.config.device))

    def save(self, path: str):
        self.attempt_save(self.hopfield_scale, path, "hopfield_scale.pt")
        super().save(path)

    def load(self, path: str):
        self.hopfield_scale = self.attempt_load(path, "hopfield_scale.pt")
        super().load(path)

    def clamp(self):
        with no_grad():
            self.logit_scale.clamp(self.clamp_min, self.clamp_max)
            self.hopfield_scale.clamp(self.clamp_min, self.clamp_max)

    def cloob(
        self,
        image_features: TensorType[-1, "latent_dim"],
        text_features: TensorType[-1, "latent_dim"],
    ) -> TensorType[(), float]:

        p_xx, p_yy, p_xy, p_yx = hopfield_retrieval(
            image_features, text_features, self.hopfield_scale
        )
        identity = torch.eye(p_xx.shape[1]) > 0.5
        i = identity.to(p_xx.device)
        loss_img = infoLOOB_loss(p_xx.T, p_xy.T, i, logit_scale=self.logit_scale)
        loss_txt = infoLOOB_loss(p_yy.T, p_yx.T, i, logit_scale=self.logit_scale)

        return (loss_img + loss_txt).sum()

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
class CARPCloobTrainer(BaseTrainer):
    def train_deepspeed_step(
        self,
        passages: BatchElement,
        reviews: BatchElement,
        config: TrainConfig,
    ):
        with self.autocast():
            forward_output = self.model(passages, reviews, config)

        # Encode passages in microbatches (with grad)
        for index, passage in enumerate(forward_output["pass_mbs"]):
            pass_tmp = forward_output["pass_encs"].copy()
            with self.autocast():
                pass_tmp[index] = self.model.module.encode_passages(passage).hidden

            loss = self.model.module.cloob(
                torch.cat(pass_tmp), torch.cat(forward_output["rev_encs"])
            )
            self.deepspeed_backwards(loss)

        # Encode reviews in microbatches (with grad)
        for index, review in enumerate(forward_output["rev_mbs"]):
            rev_tmp = forward_output["rev_encs"].copy()  # no_grad
            with self.autocast():
                rev_tmp[index] = self.model.module.encode_reviews(review).hidden
                # grad _just_ at positions in `index`
            loss = self.model.module.cloob(
                torch.cat(forward_output["pass_encs"]), torch.cat(rev_tmp)
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
            "Model/logit_scale": self.model.module.logit_scale.sum(),
            "Model/hopfield_scale": self.model.module.hopfield_scale.sum(),
        }

    def train_torch_step(
        self,
        passages: BatchElement,
        reviews: BatchElement,
        config: TrainConfig,
    ) -> Dict[str, TensorType[()]]:
        forward_output = self.model(passages, reviews, config)
        # Does gradient accumulation
        self.zero_grad()

        # Encode passages in microbatches (with grad)
        for index, passage in enumerate(forward_output["pass_mbs"]):
            pass_tmp = forward_output["pass_encs"].copy()
            with self.autocast():
                pass_tmp[index] = self.model.encode_passages(passage).hidden

            loss = self.model.cloob(
                torch.cat(pass_tmp), torch.cat(forward_output["rev_encs"])
            )
            
            self.torch_backwards(loss)
        # Encode reviews in microbatches (with grad)
        for index, review in enumerate(forward_output["rev_mbs"]):
            rev_tmp = forward_output["rev_encs"].copy()  # no_grad
            with self.autocast():
                rev_tmp[index] = self.model.encode_reviews(review).hidden
            # grad _just_ at positions in `index`
            loss = self.model.cloob(
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
            "Model/hopfield_scale": self.model.hopfield_scale.sum(),
        }
