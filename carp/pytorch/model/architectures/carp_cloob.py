import torch
import torch.nn.functional as F
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from carp.configs import ModelConfig, TrainConfig
from carp.pytorch.model.architectures import * 
from carp.pytorch.model.encoders import get_encoder
from carp.util import mbTokens, generate_indices
from typing import List
import numpy as np

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
    return (1/exp_logit_scale) * (positives + negatives)

def hopfield_retrieval(image_features, text_features, hopfield_scale):
    patterns_xx = hopfield(state_patterns=image_features, stored_patterns=image_features, hopfield_scale=hopfield_scale)
    patterns_yy = hopfield(state_patterns=text_features, stored_patterns=text_features, hopfield_scale=hopfield_scale)
    patterns_xy = hopfield(state_patterns=text_features, stored_patterns=image_features, hopfield_scale=hopfield_scale)
    patterns_yx = hopfield(state_patterns=image_features, stored_patterns=text_features, hopfield_scale=hopfield_scale)
    
    return patterns_xx, patterns_yy, patterns_xy, patterns_yx


def hopfield(state_patterns, stored_patterns, hopfield_scale):
    retrieved_patterns = stored_patterns.T @ nn.functional.softmax(
        hopfield_scale.exp() * stored_patterns @ state_patterns.t(), dim=0)
    # Column vectors -> dim=0 to normalize the column vectors
    retrieved_patterns = retrieved_patterns / retrieved_patterns.norm(dim=0, keepdim=True)
    return retrieved_patterns


patch_typeguard()

@typechecked
@register_architecture("CARP Cloob")
class CARPCloob(ContrastiveModel):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        encoder_class = get_encoder(config.encoder_type)
        self.passage_encoder = encoder_class(
            config.model_path, config.model_arch
        )
        self.review_encoder = encoder_class(
            config.model_path, config.model_arch
        )
        self.latent_dim = self.config.latent_dim
        self.pass_projector, self.rev_projector = self._make_projection_layers(self.config)

        self.logit_scale = torch.ones([], device=self.config.device) *\
            torch.log(torch.tensor([30], device=self.config.device, requires_grad=False))
        
        self.hopfield_scale = torch.ones([], device=self.config.device) *\
            torch.log(torch.tensor([8], device=self.config.device, requires_grad=False))

        self.clamp_min = torch.log(torch.tensor([1 / 100], device=self.config.device))
        self.clamp_max = torch.log(torch.tensor([100], device=self.config.device))

    def clamp(self):
        with torch.no_grad():
            self.logit_scale.clamp(self.clamp_min, self.clamp_max)
            self.hopfield_scale.clamp(self.clamp_min, self.clamp_max)
            
    def cloob(self,image_features : TensorType[-1, "latent_dim"], \
        text_features : TensorType[-1, "latent_dim"])  -> TensorType[(), float]:
        
        image_features = F.normalize(image_features)
        text_features = F.normalize(text_features)

        p_xx, p_yy, p_xy, p_yx = hopfield_retrieval(image_features, text_features, self.hopfield_scale)
        identity = torch.eye(p_xx.shape[1]) > 0.5
        i = identity.to(p_xx.device)
        loss_img = infoLOOB_loss(p_xx.T, p_xy.T, i, logit_scale=self.logit_scale)
        loss_txt = infoLOOB_loss(p_yy.T, p_yx.T, i, logit_scale=self.logit_scale)

        return (loss_img + loss_txt).sum()

    def train_step(
        self,
        passages: List[TensorType["batch", "N_pass"]],
        reviews: List[TensorType["batch", "N_rev"]],
        config: TrainConfig,
        opt: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
    ) -> Dict[str, TensorType[()]]:
        microbatch_inds = generate_indices(
            passages[0].shape[0], config.microbatch_size, shuffle=False
        )
        # Split tokens and masks into these microbatches
        pass_mbs: List[Tuple[mbTokens, mbTokens]] = [
            (passages[0][i], passages[1][i]) for i in microbatch_inds
        ]
        rev_mbs: List[Tuple[mbTokens, mbTokens]] = [
            (reviews[0][i], reviews[1][i]) for i in microbatch_inds
        ]
        
        
        # Initially get all encodings without grad
        pass_encs, rev_encs = self.calculate_embeddings(pass_mbs, rev_mbs)

        #compute accuracy
        forward_acc = self.compute_accuracy(torch.cat(pass_encs), torch.cat(rev_encs))

        opt.zero_grad()
        # Encode passages in microbatches (with grad)
        for index, passage in enumerate(pass_mbs):
            passage, mask = passage
            pass_tmp = pass_encs.copy()
            with torch.cuda.amp.autocast():
                pass_tmp[index] = self.encode_passages(
                    passage.to(self.device), mask.to(self.device)
                )
                loss = self.cloob(torch.cat(pass_tmp), torch.cat(rev_encs))
            
            scaler.scale(loss).backward()
        # Encode reviews in microbatches (with grad)
        for index, review in enumerate(rev_mbs):
            review, mask = review
            rev_tmp = rev_encs.copy()  # no_grad
            with torch.cuda.amp.autocast():
                rev_tmp[index] = self.encode_reviews(
                    review.to(self.device), mask.to(self.device)
                )  # grad _just_ at positions in `index`
                loss = self.cloob(torch.cat(pass_encs), torch.cat(rev_tmp))

            scaler.scale(loss).backward()
        # Clipping
        if self.config.grad_clip != -1:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(self.parameters(), config.grad_clip)

        scaler.step(opt)
        scaler.update()
        return {
            "Loss/Train": loss,
            "Acc/Forward": forward_acc,
            "Model/logit_scale": self.logit_scale.sum(),
            "Model/hopfield_scale": self.hopfield_scale.sum(),
        }
