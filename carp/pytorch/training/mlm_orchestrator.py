from typing import Tuple

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

from carp.configs import TrainConfig
from carp.pytorch.model.architectures import BaseModel
from carp.pytorch.training import BaseOrchestrator, register_orchestrator
from carp.util import get_scheduling_func


@register_orchestrator
class MLMOrchestrator(BaseOrchestrator):
    def __init__(self, config: TrainConfig):
        self.epoch_number = 0
        super().__init__(config)

    """
    def after_train_step(self, model: BaseModel, scheduler: _LRScheduler, opt: Optimizer, **kwargs)\
        -> Tuple[BaseModel, _LRScheduler, Optimizer]:

        if self.step_number >= 1000:
            # enable contrastive learning mode
            model.mlm_mode = False
            # change the gradient accum steps to 1
            model.config.grad_accum = 1
            # change to the batch size required for the contrastive learning component
            self.train_config.batch_size = 2048
            # reset the LR scheduler
            scheduler = LambdaLR(opt, get_scheduling_func(self.train_config))

        else:
            self.step_number += 1
        return model, scheduler, opt    
    """

    def on_epoch_start(
        self, model: BaseModel, scheduler: _LRScheduler, opt: Optimizer, **kwargs
    ) -> Tuple[BaseModel, _LRScheduler, Optimizer]:

        if self.epoch_number == 0:
            model.mlm_mode = True
        if self.epoch_number == 1:
            model.mlm_mode = False
            # change to the batch size required for the contrastive learning component
            self.train_config.batch_size = 2048
            # reset the LR scheduler
            scheduler = LambdaLR(opt, get_scheduling_func(self.train_config))
            # change the gradient accum steps to 1
            model.config.grad_accum = 1

        self.epoch_number += 1
        return model, scheduler, opt
