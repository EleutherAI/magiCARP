from typing import List

from carp.configs import ModelConfig
from carp.pytorch.model.architectures import *
from carp.pytorch.model.encoders import get_encoder
from carp.util import generate_indices

# TODO: DEEPSPEED SUPPORT (kevin)

patch_typeguard()


@typechecked
@register_architecture
class CARPMomentum(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        encoder_class = get_encoder(config.encoder_type)
        self.passage_encoder = encoder_class(config.model_path, config.model_arch)
        self.review_encoder = encoder_class(config.model_path, config.model_arch)
        self.latent_dim = config.latent_dim
        self.momentum = config.momentum
        self.pass_projector, self.rev_projector = self._make_projection_layers(config)
        self.logit_scale = nn.Parameter(
            torch.ones([], device=config.device)
            * torch.log(torch.tensor([1 / 0.07], device=config.device))
        )
        self.clamp_min = torch.log(torch.tensor([1 / 100], device=config.device))
        self.clamp_max = torch.log(torch.tensor([100], device=config.device))
        # Make momentum models
        self.passage_encoder_m = encoder_class[config.encoder_type](
            config.model_path, config.model_arch
        )
        self.review_encoder_m = encoder_class[config.encoder_type](
            config.model_path, config.model_arch
        )
        self.pass_projector_m, self.rev_projector_m = self._make_projection_layers(
            config
        )
        self.model_pairs = [
            [self.passage_encoder, self.passage_encoder_m],
            [self.pass_projector, self.pass_projector_m],
            [self.review_encoder, self.review_encoder_m],
            [self.rev_projector, self.rev_projector_m],
        ]
        self.copy_params()

    def momentum_pseudo_targets(
        self, pass_embeds, rev_embeds, pass_embeds_m, rev_embeds_m
    ):
        # Second half stolen shamelessly from Salesforce ALBEF: https://github.com/salesforce/ALBEF/blob/main/models/model_pretrain.py
        with torch.no_grad():
            pass_embeds = F.normalize(torch.cat(pass_embeds))
            rev_embeds = F.normalize(torch.cat(rev_embeds))
            pass_embeds_m = F.normalize(pass_embeds_m)
            rev_embeds_m = F.normalize(rev_embeds_m)
            sim_p2r_m = pass_embeds_m @ rev_embeds.T * self.logit_scale.exp()
            sim_r2p_m = rev_embeds_m @ pass_embeds.T * self.logit_scale.exp()
            sim_targets = torch.zeros(sim_p2r_m.size(), device=self.device)
            sim_targets.fill_diagonal_(1)
            sim_p2r_targets = (
                self.momentum * F.softmax(sim_p2r_m, dim=1)
                + (1 - self.momentum) * sim_targets
            )
            sim_r2p_targets = (
                self.momentum * F.softmax(sim_r2p_m, dim=1)
                + (1 - self.momentum) * sim_targets
            )
        return sim_p2r_targets, sim_r2p_targets

    def loss_fn(
        self, x: TensorType[-1, "latent_dim"], y: TensorType[-1, "latent_dim"], target
    ):
        """Momentum-relaxed contrastive loss.
        The basic idea is to make the contrastive learning loss a little more 'forgiving'. An EMA of the two CARP heads is
        maintained and used to estimate a "distribution" of reasonable probabilities for pairing each review in the batch with an
        associated passage, as opposed to the Dirac distribution with all of the weight just on the correct pairing (down the diagonal).
        Then we take a weighted average of the true target and the EMA estimated target and use compare the current trained heads
        predictions against _that_ weighted average.
        Args:
            x (Tensortype['batch_size', 'latent_dim']): CARP embedding of the passages (from the story)
            y (Tensortype['batch_size', 'latent_dim']): CARP embedding of the reviews
            target (Tensortype['batch_size', 'batch_size']): relaxed pseudo target from calculated from `momentum_pseudo_targets`

        Returns:
            Tensortype(scalar): The more forgiving loss
        """
        # Stolen shamelessly from Salesforce ALBEF: https://github.com/salesforce/ALBEF/blob/main/models/model_pretrain.py
        x = F.normalize(x)
        y = F.normalize(y)
        logits = x @ y.T * self.logit_scale.exp()
        loss = -torch.sum(F.log_softmax(logits, dim=1) * target, dim=1).mean() / 2
        return loss

    def encode_reviews_m(self, x):
        return self._embed_data(x, self.review_encoder_m, self.rev_projector_m)

    def encode_passages_m(self, x):
        return self._embed_data(x, self.passage_encoder_m, self.pass_projector_m)

    def momentum_embeddings(
        self,
        passages: Iterable[Tuple[BatchElement]],
        reviews: Iterable[Tuple[BatchElement]],
        return_only_embeddings: bool = True,
    ):
        self._momentum_update()
        with torch.no_grad(), torch.cuda.amp.autocast():
            pass_encs = [self.encode_passages_m(p) for p in passages]
            rev_encs = [self.encode_reviews_m(r) for r in reviews]

        # if we only need the embeddings, fetch them
        if return_only_embeddings:
            pass_encs = list(map(lambda x: x.hidden, pass_encs))
            rev_encs = list(map(lambda x: x.hidden, rev_encs))

        return torch.cat(pass_encs), torch.cat(rev_encs)

    @torch.no_grad()
    def copy_params(self):
        # Copied from Salesforce ALBEF: https://github.com/salesforce/ALBEF/blob/main/models/model_pretrain.py
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        # Copied from Salesforce ALBEF: https://github.com/salesforce/ALBEF/blob/main/models/model_pretrain.py
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data = param_m.data * self.momentum + param.data * (
                    1.0 - self.momentum
                )

    def train_step(
        self,
        passages: BatchElement,
        reviews: BatchElement,
        config: TrainConfig,
        opt: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
    ) -> Dict[str, TensorType[()]]:
        microbatch_inds = generate_indices(
            passages[0].shape[0], config.microbatch_size, shuffle=False
        )
        # Split tokens and masks into these microbatches
        pass_mbs: List[Tuple[BatchElement]] = [
            (passages[0][i], passages[1][i]) for i in microbatch_inds
        ]
        rev_mbs: List[Tuple[BatchElement]] = [
            (reviews[0][i], reviews[1][i]) for i in microbatch_inds
        ]
        # Initially get all encodings without grad
        pass_encs, rev_encs = self.calculate_embeddings(pass_mbs, rev_mbs)
        pass_encs_m, rev_encs_m = self.momentum_embeddings(pass_mbs, rev_mbs)
        sim_p2r_targets, sim_r2p_targets = self.momentum_pseudo_targets(
            pass_encs, rev_encs, pass_encs_m, rev_encs_m
        )
        opt.zero_grad()
        # Encode passages in microbatches (with grad)
        for index, passage in enumerate(pass_mbs):
            passage, mask = passage
            pass_tmp = pass_encs.copy()
            with torch.cuda.amp.autocast():
                pass_tmp[index] = self.encode_passages(
                    passage.to(self.device), mask.to(self.device)
                )
                con_loss, forward_acc = self.contrastive_loss(
                    torch.cat(pass_tmp), torch.cat(rev_encs)
                )
                if self.momentum > 0.0:
                    mom_loss_p2r = self.loss_fn(
                        torch.cat(pass_tmp), torch.cat(rev_encs), sim_p2r_targets
                    )
                else:
                    mom_loss_p2r = torch.zeros([])
                loss = self.momentum * mom_loss_p2r + (1 - self.momentum) * con_loss
            scaler.scale(loss).backward()
        # Encode reviews in microbatches (with grad)
        for index, review in enumerate(rev_mbs):
            review, mask = review
            rev_tmp = rev_encs.copy()  # no_grad
            with torch.cuda.amp.autocast():
                rev_tmp[index] = self.encode_reviews(
                    review.to(self.device), mask.to(self.device)
                )  # grad _just_ at positions in `index`
                con_loss, _ = self.contrastive_loss(
                    torch.cat(pass_encs), torch.cat(rev_tmp)
                )
                if self.momentum > 0.0:
                    mom_loss_r2p = self.loss_fn(
                        torch.cat(rev_tmp), torch.cat(pass_encs), sim_r2p_targets
                    )
                else:
                    mom_loss_r2p = torch.zeros([])
                loss = self.momentum * mom_loss_r2p + (1 - self.momentum) * con_loss
            scaler.scale(loss).backward()
        mom_loss = (mom_loss_r2p + mom_loss_p2r) / 2
        # Clipping
        if self.config.grad_clip != -1:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(self.parameters(), config.grad_clip)

        scaler.step(opt)
        scaler.update()
        return {
            "Loss/Contrastive": con_loss,
            "Loss/Momentum": mom_loss,
            "Loss/Train": loss,
            "Acc/Forward": forward_acc,
        }
