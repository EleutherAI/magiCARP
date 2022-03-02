from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass
class ModelConfig:
    latent_dim: int
    proj_dropout: float
    linear_projection: bool
    model_path: str  # info on HF model being used
    model_arch: str  # currently "roberta" or "neo" or "declutr" supported
    encoder_type: str  # "sum", "eot", "multicls", "direct" (for declutr)
    momentum: float = 0.0
    device: str = "cuda:0"
    grad_accum: int = 1
    model_eps: float = 1.0e-4  # Epsilon to add to logits in contrastive loss
    grad_clip: float = -1  # What to clip grad norms to (set to -1 for no clip)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)


@dataclass
class TrainConfig:
    n_ctx: int
    epochs: int
    batch_size: int
    microbatch_size: int
    lr_ramp_steps: int
    lr_decay_steps: int
    learning_rate_init: float
    learning_rate_target: float
    do_log: bool  # Log to WANDB?
    log_interval: int
    checkpoint_interval: int
    validate_interval: int
    eval_selection: str
    data_pipeline: str = "BaseDataPipeline"
    trainer: str = "CARPTrainer"
    # Dataset sometimes contains short reviews like "lol"
    # These are harmful during training because if a batch contains more than one
    # then the duplicates will create exploding gradients through CE loss
    # This removes all pass/rev where |pass| < 8 or |rev| < 8 (in chars)
    dupe_protection: bool = True
    hard_dupe_protection: bool = False  # Manually checks all batches for duplicates
    validation_size: int = 1000
    use_half: bool = False
    use_bucket: bool = False
    opt_eps: float = 1e-4  # Epsilon for optimizer
    weight_decay: float = 0
    gradient_checkpointing: bool = False
    gradient_averaging: bool = False

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)


@dataclass
class CARPConfig:
    model: ModelConfig
    train_job: TrainConfig

    @classmethod
    def load_yaml(cls, yml_fp: str):
        with open(yml_fp, mode="r") as file:
            config = yaml.safe_load(file)
        return cls(
            ModelConfig.from_dict(config["model"]),
            TrainConfig.from_dict(config["train_job"]),
        )

    def to_dict(self):
        data = self.model.__dict__.copy()
        data.update(self.train_job.__dict__)
        return data
