from __future__ import annotations

import sys
from typing import Any, Dict, Tuple, Iterable, Callable
from abc import abstractmethod

from datasets import load_from_disk
from torch.utils.data import Dataset
from carp.pytorch.data.utils.data_util import create_tok, TokMaskTuplePass, TokMaskTupleRev
from typeguard import typechecked
from functools import partial

from carp.configs import TrainConfig
from carp.pytorch.model.architectures import BaseModel
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
# specifies a dictionary of architectures
_ORCHESTRATORS: Dict[str, any] = {}  # registry

def register_orchestrator(name):
    """Decorator used register a CARP architecture 

        Args:
            name: Name of the architecture
    """

    def register_class(cls, name):
        _ORCHESTRATORS[name] = cls
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

# eventually orchestrator will interface with ray for distributed systems
@register_orchestrator
class BaseOrchestrator:

    def __init__(self, config : TrainConfig):
        self.config = config
    
    # if the child class does not override a trigger, just ignore it
    #TODO: We probably need way more kinds of interrupts. I dont see a way to handle this besides hand coding each though 
    @abstractmethod
    def before_validate_step(self,
        model : BaseModel,
        scheduler : _LRScheduler,
        opt : Optimizer, **kwargs) ->\
            Tuple[BaseModel, _LRScheduler, Optimizer]:
        return model, scheduler, opt

    @abstractmethod
    def after_validate_step(self,
        model : BaseModel,
        scheduler : _LRScheduler,
        opt : Optimizer, **kwargs) ->\
            Tuple[BaseModel, _LRScheduler, Optimizer]:
        return model, scheduler, opt

    @abstractmethod
    def before_train_step(self,
        model : BaseModel,
        scheduler : _LRScheduler,
        opt : Optimizer, **kwargs) ->\
            Tuple[BaseModel, _LRScheduler, Optimizer]:
        return model, scheduler, opt

    @abstractmethod
    def after_train_step(self,
        model : BaseModel,
        scheduler : _LRScheduler,
        opt : Optimizer, **kwargs) ->\
            Tuple[BaseModel, _LRScheduler, Optimizer]:
        return model, scheduler, opt

    @abstractmethod
    def before_save(self,
        model : BaseModel,
        scheduler : _LRScheduler,
        opt : Optimizer, **kwargs) ->\
            Tuple[BaseModel, _LRScheduler, Optimizer]:
        return model, scheduler, opt

    @abstractmethod
    def after_save(self,
        model : BaseModel,
        scheduler : _LRScheduler,
        opt : Optimizer, **kwargs) ->\
            Tuple[BaseModel, _LRScheduler, Optimizer]:
        return model, scheduler, opt

def get_orchestrator(name):
    return _ORCHESTRATORS[name.lower()]

def get_orchestrator_names():
    return _ORCHESTRATORS.keys()

