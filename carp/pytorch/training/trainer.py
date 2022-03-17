from __future__ import annotations

import sys
from abc import abstractmethod
from typing import Any, Callable, Dict, Tuple

import torch
import torch.distributed as dist
from catalyst.data import DistributedSamplerWrapper
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from carp.configs import TrainConfig
from carp.pytorch.data import BaseDataPipeline, get_datapipeline
from carp.pytorch.data.utils.data_util import chunkBatchElement
from carp.pytorch.model.encoders import BaseEncoder

# specifies a dictionary of architectures
_TRAINERS: Dict[str, any] = {}  # registry


def register_trainer(name):
    """Decorator used register a CARP architecture

    Args:
        name: Name of the architecture
    """

    def register_class(cls, name):
        _TRAINERS[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)

    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls


# this class handles training routine specific interrupts.
# used for methods with weird forms of learning rate schedulers, weird training routines,
# or for early stopping in hyper parameter sweeps

# eventually trainer will interface with ray for distributed systems
class BaseTrainer(object):
    def __init__(self, train_config: TrainConfig):
        self.train_config = train_config
        self.force_break = False

        # Used in determining the denominator for averaging gradients
        self.backwards_steps_cur = 0
        self.backwards_steps_max = -1

    def contrastive_parallel_all_gather(self, encs):
        encs = torch.cat(encs)
        all_encs_across_gpus = [
            torch.empty_like(encs) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(all_encs_across_gpus, encs)
        offset = dist.get_rank() * encs.size(0) // self.train_config.microbatch_size
        return encs, torch.cat(all_encs_across_gpus), offset

    def set_train_params(self, model, opt, scaler, use_deepspeed=False):
        """
        Called after optimizer and model are initialized. Not to be overridden!
        Args:
            model: Model we are training
            optimizer: Optimizer for corresponding model
            scaler: AMP grad scaler
            use_deespeed: Whether or not the trainer should use deepspeed
        """
        self.model = model
        self.opt = opt
        self.scaler = scaler
        self.use_deepspeed = use_deepspeed

    def train_step(self, *args, **kwargs):
        if self.use_deepspeed:
            return self.train_deepspeed_step(*args, **kwargs)
        else:
            return self.train_torch_step(*args, **kwargs)

    def train_deepspeed_step(self, *args, **kwargs):
        raise NotImplementedError

    def train_torch_step(self, *args, **kwargs):
        raise NotImplementedError

    def torch_backwards(self, loss):
        """
        Runs backwards using the CUDA AMP scaler
        Args:
            loss: Loss to run backwards on
        """
        self.scaler.scale(loss).backward()

        # Track the number of backwards per step
        self.backwards_steps_cur += 1
        self.backwards_steps_max = max(
            self.backwards_steps_max, self.backwards_steps_cur
        )

    def deepspeed_backwards(self, loss):
        """
        Runs backwards using the deespeed optimizer
        Args:
            loss: Loss to run backwards on
        """
        self.model.backward(loss)

        # Track the number of backwards per step
        self.backwards_steps_cur += 1
        self.backwards_steps_max = max(
            self.backwards_steps_max, self.backwards_steps_cur
        )

    def torch_step(self):
        """
        Executes a single step using the pytorch optimizer.
        """
        if self.model.accum_step % self.model.config.grad_accum == 0:
            self.scaler.step(self.opt)
            self.scaler.update()
            self.model.accum_step = 0
        else:
            self.model.accum_step += 1

    def deepspeed_step(self):
        """
        Executes a single step utilizing the deepspeed optimizer.
        """
        if self.model.module.accum_step % self.model.module.config.grad_accum == 0:
            self.model.step()
            self.model.module.accum_step = 0
        else:
            self.model.module.accum_step += 1

    def zero_grad(self):
        """
        Used to account for gradient accumulations. Accounts for deepspeed accumulation being different.
        """
        if not self.use_deepspeed:
            if self.model.accum_step % self.model.config.grad_accum == 0:
                self.opt.zero_grad()

    def average_gradients(self, steps: float = None):
        """
        Divides the model gradients by step
        Args:
            step: The denominator to divide the gradients by
        """
        # If steps is not passed, just used the estimated number of steps
        self.backwards_steps_cur = 0
        if steps is None:
            steps = self.backwards_steps_max

        # Average gradients
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad /= float(steps) ** (0.5)

    def clip_gradients(self):
        """
        Clips the model gradients as according to the train config
        """
        if self.model.config.grad_clip != -1:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.train_config.grad_clip
            )

    def eval_step(self, dataset):
        """
        Runs a single evaluation step on the model.
        Args:
            dataset: the validation dataset
        Returns:
            dict: Dictionary of validation loss and validation accuracy
        """
        passages = []
        reviews = []
        for p, r in dataset:
            passages.append(p)
            reviews.append(r)

        # TODO: Ideally should get microbatch size from trainconfig for the second argument
        passages = chunkBatchElement(passages[0], 8)
        reviews = chunkBatchElement(reviews[0], 8)

        with torch.no_grad():
            if self.use_deepspeed:
                pass_emb, rev_emb = self.model.module.calculate_embeddings(
                    passages, reviews
                )
                val_loss = self.model.module.contrastive_loss(
                    torch.cat(pass_emb), torch.cat(rev_emb)
                )
                val_acc = self.model.module.compute_accuracy(
                    torch.cat(pass_emb), torch.cat(rev_emb)
                )
            else:
                pass_emb, rev_emb = self.model.calculate_embeddings(passages, reviews)
                val_loss = self.model.contrastive_loss(
                    torch.cat(pass_emb), torch.cat(rev_emb)
                )
                val_acc = self.model.compute_accuracy(
                    torch.cat(pass_emb), torch.cat(rev_emb)
                )

        return {"Loss/Validation": val_loss.item(), "Acc/Validation": val_acc.item()}

    # if the child class does not override a trigger, just ignore it
    # TODO: We probably need way more kinds of interrupts. I dont see a way to handle this besides hand coding each though
    @abstractmethod
    def before_validate_step(self):
        pass

    @abstractmethod
    def after_validate_step(self):
        pass

    @abstractmethod
    def before_train_step(self):
        pass

    @abstractmethod
    def after_train_step(self):
        pass

    @abstractmethod
    def before_save(self):
        pass

    @abstractmethod
    def after_save(self):
        pass

    @abstractmethod
    def on_epoch_start(self):
        pass

    def construct_dataloader(
        self,
        dataset: BaseDataPipeline,
        tokenizer: Callable,
        multi_gpus: bool,
        is_train: bool,
    ) -> DataLoader:
        sampler = RandomSampler(dataset)

        if multi_gpus is True:
            sampler = DistributedSamplerWrapper(
                sampler=sampler,
                shuffle=False,
            )

        return DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            sampler=sampler,
            collate_fn=tokenizer,
            drop_last=True if self.use_deepspeed and is_train else False,
        )

    def construct_tokenizer(self, passage_encoder: BaseEncoder) -> Callable:
        call_tokenizer = passage_encoder.call_tokenizer
        tokenizer_factory = get_datapipeline(
            self.train_config.data_pipeline
        ).tokenizer_factory
        tokenizer = get_datapipeline(
            self.train_config.data_pipeline
        ).create_tokenizer_factory(
            call_tokenizer, tokenizer_factory, self.train_config.n_ctx
        )
        return tokenizer(passage_encoder)


from carp.pytorch.model.architectures.carp import CARPTrainer
from carp.pytorch.model.architectures.carp_cloob import CARPCloobTrainer
from carp.pytorch.model.architectures.carp_coop import CARPCoOpTrainer

# from carp.pytorch.model.architectures.carp_mlm import CARPMLM
# from carp.pytorch.model.architectures.carp_momentum import CARPMomentum
from carp.pytorch.model.architectures.carp_shared_encoder import (
    CARPSharedEncoderTrainer,
)


def get_trainer(name):
    return _TRAINERS[name.lower()]


def get_trainer_names():
    return _TRAINERS.keys()
