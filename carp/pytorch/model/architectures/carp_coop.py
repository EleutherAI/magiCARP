import os
from typing import List

import numpy as np

from carp.configs import ModelConfig
from carp.pytorch.data.scarecrow_pipeline import ScarecrowTargetElement
from carp.pytorch.model.architectures import *
from carp.pytorch.model.encoders import BaseEncoder, BaseEncoderOutput
from carp.pytorch.training import BaseTrainer, register_trainer
from carp.util import generate_indices

os.environ["TOKENIZERS_PARALLELISM"] = "true"


# Written by MicPie
class PromptLayer(nn.Module):
    def __init__(
        self,
        encoder: BaseEncoder,
        labels: List[str] = None,
        n_ctx: int = 10,
        ctx_dim: int = 1024,
    ):
        super().__init__()
        if labels is None:
            labels = [
                "Off-prompt",
                "Grammar Usage",
                "Needs Google",
                "Incoherent",
                "Technical Jargon",
                "Redundant",
            ]

        self.labels = labels
        self.n_labels = len(labels)

        # Random init n_ctx vectors of dim ctx_dim,
        # can also initialize context vectors to something else like in the paper.
        ctx_vectors = torch.empty(n_ctx, ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02)  # std = 0.02 from appendix A
        self.ctx = nn.Parameter(ctx_vectors)  # learnable
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim

        toks = [encoder.call_tokenizer(l) for l in labels]
        with torch.no_grad():  # not needed?
            toks_embs_masks = [
                {
                    "input_ids": t["input_ids"],
                    "input_embs": encoder.model.embeddings(t["input_ids"]),
                    "attention_mask": t["attention_mask"],
                }
                for t in toks
            ]
        del toks

        seq_len_max = np.max([e["input_embs"].shape[1] for e in toks_embs_masks])

        self.toks = torch.zeros(self.n_labels, seq_len_max, dtype=torch.int)
        embs = torch.zeros(self.n_labels, seq_len_max, ctx_dim, dtype=torch.float)
        masks = torch.zeros(self.n_labels, seq_len_max, dtype=torch.bool)

        for i, e in enumerate(toks_embs_masks):
            self.toks[i, : e["input_ids"].shape[1]] = e["input_ids"]
            embs[i, : e["input_embs"].shape[1]] = e["input_embs"].squeeze(0)
            masks[i, : e["attention_mask"].shape[1]] = e["attention_mask"]

        self.prefix_embs = embs[:, :1, :].cuda()  # sos
        self.suffix_embs = embs[:, 1:, :].cuda()  # cls, eos
        self.prefix_masks = masks[:, :1].cuda()  # sos
        self.suffix_masks = masks[:, 1:].cuda()  # cls, eos

        self.prefix_embs.requires_grad = True
        self.suffix_embs.requires_grad = True

    def forward(self):
        ctx = self.ctx
        if (
            ctx.dim() == 2
        ):  # With the current setup this is always needed! TO DO: Check!
            ctx = ctx.unsqueeze(0).expand(self.n_labels, self.n_ctx, self.ctx_dim)

        prompts = torch.cat(
            [
                self.prefix_embs,  # [n_labels, 1,     dim]
                ctx,  # [n_labels, n_ctx, dim]
                self.suffix_embs,  # [n_labels, *,     dim]
            ],
            dim=1,
        )

        masks = torch.cat(
            [
                self.prefix_masks,  # [n_labels, 1]
                torch.ones((self.n_labels, self.n_ctx)).cuda(),  # [n_labels, n_ctx]
                self.suffix_masks,  # [n_labels, *]
            ],
            dim=1,
        )

        return prompts, masks


# CARP CoOp uses prompt tuning as a method to boost biencoder contrastive learning models.
#  Useful for downstream tasks
# TODO: Custom training routine. Might need to abstract training significantly
patch_typeguard()


@typechecked
@register_architecture
class CARPCoOp(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
        self.review_encoder_CoOp = PromptLayer(self.review_encoder)

        # required for CoOp
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

    def save(self, path: str):
        torch.save(self.review_encoder_CoOp, path + "review_encoder_CoOp.pt")
        super().save(path)

    def load(self, path: str):
        try:
            self.review_encoder_CoOp = torch.load(path + "review_encoder_CoOp.pt")
        except:
            print(
                "Unable to load review_encoder_CoOp. Randomly initializing and continuing."
            )
        super().load(path)

    # uses a constant set of reviews for CoOp
    def calculate_embeddings(
        self,
        passages: Iterable[Tuple[BatchElement]],
        return_only_embeddings: bool = True,
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
        y_CoOp, mask_CoOp = self.review_encoder_CoOp()
        y_CoOp = self.review_encoder(y_CoOp, mask_CoOp, inputs_embeds=True)
        return BaseEncoderOutput(self.rev_projector(y_CoOp.hidden))

    def compute_accuracy(
        self,
        x: TensorType["pass_N", "latent_dim"],
        y: TensorType[-1, "latent_dim"],
        labels: TensorType["pass_N", -1],
    ):
        """Computes KL divergence against target CoOp distribution

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
            acc = (torch.argmax(logits, dim=1) == labels).sum()
        return acc / x.shape[0]

    def CoOp_loss(
        self,
        x: TensorType["pass_N", "latent_dim"],
        y: TensorType[-1, "latent_dim"],
        labels: TensorType["pass_N", -1],
    ):
        """Computes KL divergence against target CoOp distribution

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
        return F.kl_div(logits.float(), labels.float(), reduction="batchmean")

    def eval_step(self, dataset):
        passages = []
        reviews = []
        for p, r in dataset:
            passages.append(p)
            reviews.append(r)

        # TODO: Ideally should get microbatch size from trainconfig for the second argument
        passages = chunkBatchElement(passages[0], 8)
        rev_labels = torch.cat(
            list(map(lambda x: x.target_dist.cuda(), reviews)), dim=0
        )

        with torch.no_grad():
            pass_emb, rev_emb = self.calculate_embeddings(passages)
            val_loss = self.CoOp_loss(torch.cat(pass_emb), rev_emb, rev_labels)
            val_acc = self.compute_accuracy(torch.cat(pass_emb), rev_emb, rev_labels)

        return {"Loss/Validation": val_loss.item(), "Acc/Validation": val_acc.item()}

    def forward(
        self,
        passages: BatchElement,
        reviews: ScarecrowTargetElement,
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
        # create array of rev_labels and cast to GPU
        rev_labels: List[torch.tensor] = [
            reviews.target_dist[i].cuda() for i in microbatch_inds
        ]

        # Initially get all encodings without grad
        pass_encs, rev_encs = self.calculate_embeddings(pass_mbs)

        # compute accuracy. We need labels for CoOp accuracy
        forward_acc = self.compute_accuracy(
            torch.cat(pass_encs), rev_encs, torch.cat(rev_labels)
        )

        return {
            "pass_mbs": pass_mbs,
            "pass_encs": pass_encs,
            "rev_encs": rev_encs,
            "forward_acc": forward_acc,
        }

@register_trainer
class CARPCoOpTrainer(BaseTrainer):
    def train_deepspeed_step(
        self,
        passages: BatchElement,
        reviews: ScarecrowTargetElement,
        config: TrainConfig,
    ) -> Dict[str, TensorType[()]]:
        forward_output = self.model(passages, reviews, config)

        # Encode passages in microbatches (with grad) and compute CoOp loss
        for index, passage in enumerate(forward_output["pass_mbs"]):
            pass_tmp = forward_output["pass_encs"].copy()
            with torch.cuda.amp.autocast():
                pass_tmp[index] = self.model.module.encode_passages(passage).hidden

            loss = self.model.module.CoOp_loss(
                torch.cat(pass_tmp),
                forward_output["rev_encs"],
                torch.cat(forward_output["rev_labels"]),
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
        reviews: ScarecrowTargetElement,
        config: TrainConfig,
    ) -> Dict[str, TensorType[()]]:
        forward_output = self.model(passages, reviews, config)

        # does gradient accumulation
        self.zero_grad(self.opt)

        # Encode passages in microbatches (with grad) and compute CoOp loss
        for index, passage in enumerate(forward_output["pass_mbs"]):
            pass_tmp = forward_output["pass_encs"].copy()
            with torch.cuda.amp.autocast():
                pass_tmp[index] = self.model.encode_passages(passage).hidden
                loss = self.model.CoOp_loss(
                    torch.cat(pass_tmp),
                    forward_output["rev_encs"],
                    torch.cat(forward_output["rev_labels"]),
                )

            self.scaler.scale(loss).backward()

        # Clipping
        if self.model.config.grad_clip != -1:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip)

        self.torch_step()

        return {
            "Loss/Train": loss,
            "Acc/Forward": forward_output["forward_acc"],
        }
