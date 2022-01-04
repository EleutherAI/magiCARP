from carp.configs import TrainConfig
from carp.pytorch.training import BaseOrchestrator, register_orchestrator

from carp.configs import TrainConfig
from carp.pytorch.model.architectures import BaseModel
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from torch.optim import Optimizer

from typing import Tuple

from carp.util import get_scheduling_func


@register_orchestrator
class MLMOrchestrator(BaseOrchestrator):
    def __init__(self, config: TrainConfig):
        self.step_number = 0
        super().__init__(config)
    def after_train_step(self, model: BaseModel, scheduler: _LRScheduler, opt: Optimizer, **kwargs)\
        -> Tuple[BaseModel, _LRScheduler, Optimizer]:

        if self.step_number == 20:
            model.mlm_mode = False
            # change to the batch size required for the contrastive learning component
            self.train_config.batch_size = 2048
            # reset the LR scheduler
            scheduler = LambdaLR(opt, get_scheduling_func(self.train_config))
            # change the gradient accum steps to 1
            model.config.grad_accum = 1
        else:
            self.step_number += 1
        return model, scheduler, opt
