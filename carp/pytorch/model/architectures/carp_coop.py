import torch
import torch.nn.functional as F
import numpy as np
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from dataclasses import dataclass
from carp.configs import CARPConfig, ModelConfig, TrainConfig
from carp.pytorch.data.scarecrow_pipeline import BatchElement, ScarecrowTargetElement
from carp.pytorch.model.architectures import * 
from carp.pytorch.model.encoders import BaseEncoder, get_encoder
from carp.util import generate_indices
from typing import List

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"



# Written by MicPie
class PromptLayer(nn.Module):
    def __init__(self, encoder : BaseEncoder,
        labels : List[str] = 
        ['Off-prompt', 'Grammar Usage',
        'Needs Google', 'Incoherent', 
        'Technical Jargon', 'Redundant'], 
        n_ctx : int = 10, ctx_dim : int = 512):

        super().__init__()

        self.labels = labels
        self.n_labels = len(labels)

        # Random init n_ctx vectors of dim ctx_dim,
        # can also initialize context vectors to something else like in the paper.
        ctx_vectors = torch.empty(n_ctx, ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02) # std = 0.02 from appendix A
        self.ctx = nn.Parameter(ctx_vectors) # learnable
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim

        toks = [encoder.call_tokenizer(l) for l in labels]
        with torch.no_grad(): # not needed?
            toks_embs_masks = [{
                'input_ids': t['input_ids'],
                'input_embs': encoder.model.embeddings(t['input_ids']),
                'attention_mask': t['attention_mask']}
            for t in toks]
        del toks

        seq_len_max = np.max([e['input_embs'].shape[1] for e in toks_embs_masks])

        self.toks = torch.zeros(self.n_labels, seq_len_max, dtype=torch.int)
        embs = torch.zeros(self.n_labels, seq_len_max, ctx_dim, dtype=torch.float)
        masks = torch.zeros(self.n_labels, seq_len_max, dtype=torch.bool)

        for i, e in enumerate(toks_embs_masks):
            self.toks[i,:e['input_ids'].shape[1]] = e['input_ids']
            embs[i,:e['input_embs'].shape[1]] = e['input_embs'].squeeze(0)
            masks[i,:e['attention_mask'].shape[1]] = e['attention_mask']

        self.prefix_embs = embs[:, :1, :].cuda() # sos
        self.suffix_embs = embs[:, 1:, :].cuda() # cls, eos
        self.prefix_masks = masks[:, :1].cuda() # sos
        self.suffix_masks = masks[:, 1:].cuda() # cls, eos

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2: # With the current setup this is always needed! TO DO: Check!
            ctx = ctx.unsqueeze(0).expand(self.n_labels, self.n_ctx, self.ctx_dim)

        prompts = torch.cat([
            self.prefix_embs, # [n_labels, 1,     dim]
            ctx,              # [n_labels, n_ctx, dim]
            self.suffix_embs  # [n_labels, *,     dim]
        ], dim=1)

        masks = torch.cat([
            self.prefix_masks,  # [n_labels, 1]
            torch.ones((self.n_labels, self.n_ctx)).cuda(), # [n_labels, n_ctx]
            self.suffix_masks   # [n_labels, *]
        ], dim=1)

        return prompts, masks


# CARP CoOP uses prompt tuning as a method to boost biencoder contrastive learning models.
#  Useful for downstream tasks
# TODO: Custom training routine. Might need to abstract training significantly
patch_typeguard()

@typechecked
@register_architecture
class CARPCoOP(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
        encoder_class = get_encoder(config.encoder_type)
        self.passage_encoder = encoder_class(
            config.model_path, config.model_arch
        )
        self.review_encoder = encoder_class(
            config.model_path, config.model_arch
        )
        self.review_encoder_coop = PromptLayer(self.review_encoder)

        self.latent_dim = self.config.latent_dim
        self.pass_projector, self.rev_projector = self._make_projection_layers(self.config)
        self.logit_scale = nn.Parameter(
            torch.ones([], device=self.config.device)
            * torch.log(torch.tensor([1 / 0.07], device=self.config.device))
        )
        self.clamp_min = torch.log(torch.tensor([1 / 100], device=self.config.device))
        self.clamp_max = torch.log(torch.tensor([100], device=self.config.device))

        # required for CoOP
        self.freeze_encoders()
    # freezes encoder and projection layer
    def freeze_encoders(self):
        for params in self.passage_encoder.parameters():
            params.requires_grad_(False)
        for params in self.review_encoder.parameters():
            params.requires_grad_(False)
        for params in self.pass_projector.parameters():
            params.requires_grad_(False)
        for params in self.rev_projector.parameters():
            params.requires_grad_(False)

    # uses a constant set of reviews for CoOP
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
            rev_encs = self.encode_reviews()
        
        # if we only need the embeddings, fetch them
        if return_only_embeddings:
            pass_encs = list(map(lambda x: x.hidden, pass_encs))
            rev_encs = list(map(lambda x: x.hidden, rev_encs))
        return pass_encs, rev_encs

    def encode_reviews(self):
        y_coop, mask_coop = self.review_encoder_coop()
        y_coop = self.review_encoder(y_coop, mask_coop, inputs_embeds=True)
        return self.rev_projector(y_coop)

    def train_step(
        self,
        passages: BatchElement,
        reviews: ScarecrowTargetElement,
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
        rev_mbs: List[ScarecrowTargetElement] = [
            ScarecrowTargetElement(reviews.target_dist[i]) for i in microbatch_inds
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
