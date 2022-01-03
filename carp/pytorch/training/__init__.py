from __future__ import annotations

import sys
from typing import Callable, Dict, Tuple
from abc import abstractmethod

from torch.utils.data.sampler import RandomSampler

from carp.configs import TrainConfig
from carp.pytorch.data import BaseDataPipeline, get_datapipeline
from carp.pytorch.model.architectures import BaseModel
from carp.pytorch.model.encoders import BaseEncoder
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from torch.utils.data import DataLoader

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

    def __init__(self, train_config : TrainConfig):
        self.train_config = train_config
    
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

    @abstractmethod
    def on_epoch_start(self,
        model : BaseModel,
        scheduler : _LRScheduler,
        opt : Optimizer, **kwargs) ->\
            Tuple[BaseModel, _LRScheduler, Optimizer]:
        return model, scheduler, opt
    
    def construct_dataloader(self,
        dataset : BaseDataPipeline,
        tokenizer : Callable) -> DataLoader:
        sampler = RandomSampler(dataset)
        return DataLoader(
                    dataset,
                    batch_size=self.train_config.microbatch_size,
                    sampler=sampler,
                    collate_fn=tokenizer,
                )
    def construct_tokenizer(self,
        passage_encoder : BaseEncoder) -> Callable:
        call_tokenizer = passage_encoder.call_tokenizer
        tokenizer_factory = get_datapipeline(self.train_config.data_pipeline).tokenizer_factory
        tokenizer =\
            get_datapipeline(self.train_config.data_pipeline).create_tokenizer_factory(
                call_tokenizer,
                tokenizer_factory,
                self.train_config.n_ctx)
        return tokenizer(passage_encoder)        


from carp.pytorch.training.mlm_orchestrator import MLMOrchestrator

def get_orchestrator(name):
    return _ORCHESTRATORS[name.lower()]

def get_orchestrator_names():
    return _ORCHESTRATORS.keys()

