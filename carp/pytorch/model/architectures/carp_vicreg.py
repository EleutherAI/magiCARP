from typing import List

import torch.distributed as dist

from carp.configs import ModelConfig
from carp.pytorch.model.architectures import *
from carp.pytorch.scalability_utils import print_rank_0
from carp.pytorch.training.trainer import BaseTrainer, register_trainer
from carp.util import generate_indices


def off_diagonal(x):
    """
    Returns the off-diagonal elements of a 2D tensor.
    :param x: A 2D tensor.
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def variance_penalty(x: TensorType[-1, "latent_dim"], epsilon=1e-4):
    """
    Variance penalty for the encodings.
    :param encodings: The encodings to apply the penalty to.
    :param epsilon: The epsilon to use for the penalty.
    :return: The penalty.
    """
    # subtract mean
    x_mean = torch.mean(x, dim=0)
    x_ = x - x_mean

    # Calculate the variance of one set of encodings.
    std_x = torch.sqrt(torch.var(x_, dim=0) + epsilon)
    return torch.mean(F.relu(1 - std_x))


def covariance_penalty(x: TensorType[-1, "latent_dim"]):
    """
    Covariance penalty for the encodings.
    :param encodings: The encodings to apply the penalty to.
    :param epsilon: The epsilon to use for the penalty.
    """

    # subtract mean
    x_mean = torch.mean(x, dim=0)
    x_ = x - x_mean

    # Calculate the covariance of the encodings
    cov = torch.matmul(x_.t(), x_) / x_.shape[0]
    # Calculate the covariance penalty
    return off_diagonal(cov).pow_(2).sum() / (x_.shape[1])


def vicreg_penalty(encodings: TensorType[-1, "latent_dim"], epsilon=1e-4):
    """
    Computes the penalty for the encodings.
    :param encodings: The encodings to apply the penalty to.
    :param epsilon: The epsilon to use for the penalty.
    :return: The penalty.
    """
    return variance_penalty(encodings, epsilon=epsilon) + covariance_penalty(encodings)


patch_typeguard()


@register_trainer
class CARPVicregTrainer(BaseTrainer):
    def train_deepspeed_step(
        self,
        passages: BatchElement,
        reviews: BatchElement,
        config: TrainConfig,
    ):
        with self.autocast():
            forward_output = self.model(passages, reviews, config)

        _, all_rev_encs, rev_offset = self.contrastive_parallel_all_gather(
            forward_output["rev_encs"]
        )
        _, all_pass_encs, pass_offset = self.contrastive_parallel_all_gather(
            forward_output["pass_encs"]
        )

        # Encode passages in microbatches (with grad)
        for index, passage in enumerate(forward_output["pass_mbs"]):
            pass_tmp = list(all_pass_encs.clone().split(config.microbatch_size))
            with self.autocast():
                micro_batch = self.model.module.encode_passages(passage).hidden
                pass_tmp[pass_offset + index] = micro_batch
                loss = self.model.module.contrastive_loss(
                    torch.cat(pass_tmp), all_rev_encs
                ) + (
                    self.model.module.penalty(torch.cat(pass_tmp))
                    / len(forward_output["pass_mbs"])
                )

            self.deepspeed_backwards(loss)

        # Encode reviews in microbatches (with grad)
        for index, review in enumerate(forward_output["rev_mbs"]):
            rev_tmp = list(all_rev_encs.clone().split(config.microbatch_size))
            with self.autocast():
                micro_batch = self.model.module.encode_reviews(review).hidden
                rev_tmp[rev_offset + index] = micro_batch
                loss = self.model.module.contrastive_loss(
                    all_pass_encs, torch.cat(rev_tmp)
                ) + (
                    self.model.module.penalty(torch.cat(rev_tmp))
                    / len(forward_output["rev_mbs"])
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
        }

    def train_torch_step(
        self,
        passages: BatchElement,
        reviews: BatchElement,
        config: TrainConfig,
    ) -> Dict[str, TensorType[()]]:
        with self.autocast():
            forward_output = self.model(passages, reviews, config)

        # does gradient accumulation
        self.zero_grad()
        vicreg_loss = None
        # Encode passages in microbatches (with grad)
        for index, passage in enumerate(forward_output["pass_mbs"]):
            pass_tmp = forward_output["pass_encs"].copy()
            with self.autocast():
                pass_tmp[index] = self.model.encode_passages(passage).hidden
                loss = self.model.contrastive_loss(
                    torch.cat(pass_tmp), torch.cat(forward_output["rev_encs"])
                )
                penalty = vicreg_penalty(torch.cat(pass_tmp)) / len(
                    forward_output["pass_mbs"]
                )
                if vicreg_loss is None:
                    vicreg_loss = penalty.detach()
                else:
                    vicreg_loss += penalty.detach()

            self.torch_backwards(loss + penalty)

        # Encode reviews in microbatches (with grad)
        for index, review in enumerate(forward_output["rev_mbs"]):
            rev_tmp = forward_output["rev_encs"].copy()  # no_grad
            with self.autocast():
                rev_tmp[index] = self.model.encode_reviews(review).hidden
                # grad _just_ at positions in `index`
                loss = self.model.contrastive_loss(
                    torch.cat(forward_output["pass_encs"]), torch.cat(rev_tmp)
                )
                penalty = vicreg_penalty(torch.cat(rev_tmp)) / len(
                    forward_output["rev_mbs"]
                )
                vicreg_loss += penalty.detach()

            self.torch_backwards(loss + penalty)

        # Average the model gradients
        self.average_gradients()

        # Clipping
        self.clip_gradients()

        # Step the model
        self.torch_step()

        return {
            "Loss/Train": loss,
            "Loss/Vicreg": vicreg_loss,
            "Acc/Forward": forward_output["forward_acc"],
            "Acc/Top_5_Forward": forward_output["top_k_acc"],
            "Model/logit_scale": self.model.logit_scale.sum(),
        }
