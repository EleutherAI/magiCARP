import torch
import torch.nn.functional as F
import numpy as np
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from dataclasses import dataclass
from carp.configs import CARPConfig, ModelConfig, TrainConfig
from carp.pytorch.data.scarecrow_pipeline import BatchElement, ScarecrowTargetElement
from carp.pytorch.model.architectures import * 
from carp.pytorch.model.encoders import BaseEncoder, BaseEncoderOutput, get_encoder
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

        self.prefix_embs.requires_grad = True
        self.suffix_embs.requires_grad = True

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

    def save(self, path : str):
        torch.save(self.review_encoder_coop, path + "review_encoder_coop.pt")
        super().save(path)

    def load(self, path : str):
        try:
            self.review_encoder_coop = torch.load(path + "review_encoder_coop.pt")
        except:
            print("Unable to load review_encoder_coop. Randomly initializing and continuing.")

    # uses a constant set of reviews for CoOP
    def calculate_embeddings(
        self,
        passages: Iterable[
            Tuple[
                BatchElement
            ]
        ],
        return_only_embeddings : bool = True,
    ):
        # Get encodings without grad
        with torch.no_grad(), torch.cuda.amp.autocast():
            pass_encs = [self.encode_passages(p) for p in passages]

        with torch.cuda.amp.autocast():
            rev_encs = self.encode_reviews()

        # if we only need the embeddings, fetch them
        if return_only_embeddings:
            pass_encs = list(map(lambda x: x.hidden, pass_encs))
            rev_encs = rev_encs.hidden

        return pass_encs, rev_encs

    def encode_reviews(self):
        y_coop, mask_coop = self.review_encoder_coop()
        y_coop = self.review_encoder(y_coop, mask_coop, inputs_embeds=True)
        return BaseEncoderOutput(self.rev_projector(y_coop.hidden))

    def compute_accuracy(self,
        x: TensorType["pass_N", "latent_dim"],
        y: TensorType[-1, "latent_dim"],
        labels: TensorType["pass_N", -1]):
        """Computes KL divergence against target CoOP distribution

            Args:
                x: Tensor of passage encodings
                y: Tensor of review encodings, with concat soft prompt embeddings
                labels: Target distribution
            Returns:
                loss: Float (without gradient)
        """
        with torch.no_grad():
            x = F.normalize(x)
            y = F.normalize(y)
            logits = F.softmax(x @ y.T * self.logit_scale.exp(), dim=-1)
            labels = labels.argmax(1)
            acc = (torch.argmax(logits, dim = 1) == labels).sum()
        return acc/x.shape[0]

    def coop_loss(self, 
        x: TensorType["pass_N", "latent_dim"],
        y: TensorType[-1, "latent_dim"],
        labels: TensorType["pass_N", -1]):
        """Computes KL divergence against target CoOP distribution

            Args:
                x: Tensor of passage encodings
                y: Tensor of review encodings, with concat soft prompt embeddings
                labels: Target distribution
            Returns:
                loss: Float (with gradient)
        """
        x = F.normalize(x)
        y = F.normalize(y)
        logits = F.log_softmax(x @ y.T * self.logit_scale.exp(), dim=-1)
        return F.kl_div(logits.float(), labels.float(), reduction='batchmean')
    
    def eval_step(self, dataset):
        passages = []
        reviews = []
        for p, r in dataset:
            passages.append(p)
            reviews.append(r)
        
        # TODO: Ideally should get microbatch size from trainconfig for the second argument
        passages = chunkBatchElement(passages[0], 8)
        rev_labels = torch.cat(list(map(lambda x: x.target_dist.cuda(), reviews)), dim=0)

        with torch.no_grad():
            pass_emb, rev_emb = self.calculate_embeddings(passages)
            val_loss = self.coop_loss(torch.cat(pass_emb), rev_emb, rev_labels)
            val_acc = self.compute_accuracy(torch.cat(pass_emb), rev_emb, rev_labels)

        return {"Loss/Validation": val_loss.item(), "Acc/Validation": val_acc.item()}
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
        # create array of rev_labels and cast to GPU
        rev_labels: List[torch.tensor] = [
            reviews.target_dist[i].cuda() for i in microbatch_inds
        ]

        # Initially get all encodings without grad
        pass_encs, rev_encs = self.calculate_embeddings(pass_mbs)

        #compute accuracy. We need labels for CoOP accuracy
        forward_acc = self.compute_accuracy(torch.cat(pass_encs), rev_encs, torch.cat(rev_labels))

        # does gradient accumulation
        self.zero_grad(opt)

        # Encode passages in microbatches (with grad) and compute coop loss
        for index, passage in enumerate(pass_mbs):
            pass_tmp = pass_encs.copy()
            with torch.cuda.amp.autocast():
                pass_tmp[index] = self.encode_passages(passage).hidden
                loss  = self.coop_loss(
                    torch.cat(pass_tmp), rev_encs, torch.cat(rev_labels))
                    
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
