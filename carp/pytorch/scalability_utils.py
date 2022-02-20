import json

import torch.distributed as dist


def parse_deepspeed_config(args, config, lr, weight_decay, **kwargs):
    if args.deepspeed_config is not None:
        deepspeed_config = json.load(open(args.deepspeed_config))
        deepspeed_config["train_batch_size"] = config.batch_size

        assert (
            "gradient_accumulation_steps" not in deepspeed_config
        ), "Don't use 'gradient_accumulation_steps' in deepspeed config."
        assert (
            "train_micro_batch_size_per_gpu" not in deepspeed_config
        ), "Don't use 'train_micro_batch_size_per_gpu' in deepspeed config."
        assert (
            "scheduler" not in deepspeed_config
        ), "Don't use 'scheduler' in deepspeed config."
        assert (
            "optimizer" in deepspeed_config
        ), "Please specify 'optimizer' in deepspeed config."

        deepspeed_config["optimizer"]["params"]["lr"] = lr
        deepspeed_config["optimizer"]["params"]["eps"] = config.opt_eps
        deepspeed_config["optimizer"]["params"]["weight_decay"] = weight_decay
        return deepspeed_config


def init_process_group(backend):
    dist.init_process_group(backend)


def fn_rank_0(fn, *args, **kwargs):
    if dist.is_initialized() and dist.is_available():
        if dist.get_rank() == 0:
            return fn(*args, **kwargs)
    else:
        return fn(*args, **kwargs)


def print_rank_0(*args, **kwargs):
    fn_rank_0(print, *args, **kwargs)
