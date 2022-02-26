from argparse import ArgumentParser
from pathlib import Path

import deepspeed
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Subset, random_split
from transformers import PreTrainedModel

import wandb
from carp.clock import Clock
from carp.configs import CARPConfig
from carp.pytorch.data import BaseDataPipeline, get_datapipeline
from carp.pytorch.model.architectures import get_architecture
from carp.pytorch.scalability_utils import (
    fn_rank_0,
    init_process_group,
    parse_deepspeed_config,
    print_rank_0,
)
from carp.pytorch.training import get_trainer
from carp.pytorch.training.utils import print_available_configs
from carp.util import get_scheduling_func


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=False)
    parser.add_argument("--ckpt_path", type=str, default="./output/", required=False)
    parser.add_argument("--config_path", type=str, default="./base_config.yml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--wandb_run_name", type=str)
    parser.add_argument("--seed", type=float, default=42)
    parser.add_argument("--type", type=str, default="CARP")
    parser.add_argument("--get_architectures", action="store_true")
    parser.add_argument("--get_encoders", action="store_true")
    parser.add_argument("--get_datapipelines", action="store_true")
    parser.add_argument("--get_trainers", action="store_true")
    parser.add_argument("--deepspeed_config", default=None)
    return parser


def sanity_check(args, config):
    if not config.train_job.use_half:
        if args.deepspeed_config is not None and "fp16" in args.deepspeed_config:
            raise ValueError(
                "You specified fp16 in the deepspeed config, "
                "but config.use_half is False."
            )

    if args.deepspeed_config is not None and "fp16" in args.deepspeed_config:
        if not config.train_job.use_half:
            raise ValueError(
                "You specified fp16 in the deepspeed config, "
                "but config.use_half is False."
            )


def get_model(
    config: CARPConfig, load_checkpoint: bool, model_type: str = "CARP", ckpt_path=None
):
    model = get_architecture(model_type)(config.model)
    if load_checkpoint:
        model.load(ckpt_path)
        print_rank_0("Checkpoint loaded!")
    model.cuda()
    if config.train_job.use_half:
        model.half()
    if config.train_job.gradient_checkpointing:
        for module in model.modules():
            if isinstance(module, PreTrainedModel):
                module.gradient_checkpointing_enable()
    return model


def get_datasets(config, data_path, random_seed=None):
    dataset = get_datapipeline(config.data_pipeline)(config.dupe_protection, data_path)
    size = len(dataset)

    seed = torch.manual_seed(random_seed)
    if config.eval_selection == "random":
        splits = [size - config.validation_size, config.validation_size]
        return random_split(dataset, splits, generator=seed)
    elif config.eval_selection == "final_n":
        train_indices = list(range(size - config.validation_size))
        eval_indices = list(range(size - config.validation_size, size))
        return Subset(dataset, train_indices), Subset(dataset, eval_indices)
    else:
        raise NotImplementedError(
            'The only valid options for `eval_selection` are "random" and "final_n"'
        )


def save_checkpoint(save_fn, scheduler, opt, iter: int, save_iter: bool):
    # this function should be called in the fn_rank_0
    print("SAVING...")
    # Only save extra once every 20
    if save_iter:
        Path(f"./checkpoints/{iter}/").mkdir(parents=True, exist_ok=True)
        save_fn(f"./checkpoints/{iter}/")

    Path("./output/").mkdir(parents=True, exist_ok=True)
    save_fn("./output/")
    torch.save(scheduler.state_dict(), "./output/schedule.pt")
    torch.save(opt.state_dict(), "./output/opt.pt")


# Dataset assumed to be list of pairs on memory
def train(
    model,
    dataset: BaseDataPipeline,
    evalset: BaseDataPipeline,
    trainer,
    args,
    multi_gpus: bool = False,
):
    # Tokenizes string batch using encoder tokenizer
    USE_DEEPSPEED = args.deepspeed_config is not None
    LEARNING_RATE_INIT = trainer.train_config.learning_rate_init
    LOAD_CHECKPOINT = args.load_checkpoint

    # setup data pipeline. model is needed
    tokenizer = trainer.construct_tokenizer(model.passage_encoder)

    if USE_DEEPSPEED:
        model, opt, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=args.deepspeed_config,
            mpu=model.mpu if hasattr(model, "mpu") else None,
        )
        scheduler = LambdaLR(opt.optimizer, get_scheduling_func(trainer.train_config))

    else:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE_INIT,
            weight_decay=0,
            eps=trainer.train_config.opt_eps,
        )
        scheduler = LambdaLR(opt, get_scheduling_func(trainer.train_config))

    scaler = torch.cuda.amp.GradScaler()
    save_fn = model.module.save if USE_DEEPSPEED else model.save
    trainer.set_train_params(model, opt, scaler, USE_DEEPSPEED)

    if LOAD_CHECKPOINT:
        try:
            if args.ckpt_path is None:
                scheduler.load_state_dict(
                    torch.load("./output/schedule.pt", map_location="cpu")
                )
                opt.load_state_dict(torch.load("./output/opt.pt", map_location="cpu"))
            else:
                scheduler.load_state_dict(
                    torch.load(args.ckpt_path + "schedule.pt", map_location="cpu")
                )
                opt.load_state_dict(
                    torch.load(args.ckpt_path + "opt.pt", map_location="cpu")
                )
        except:
            print_rank_0("Unable to load scheduler and/or optimizer. Continuing.")

    model.train()
    timer = Clock()
    iteration, best_val = 0, 100

    for epoch in range(trainer.train_config.epochs):
        model, scheduler, opt = trainer.on_epoch_start(model, scheduler, opt)
        train_data = trainer.construct_dataloader(dataset, tokenizer, multi_gpus)

        for passages, reviews in train_data:
            timer.hit()
            model, scheduler, opt = trainer.before_train_step(model, scheduler, opt)
            batch_outputs = trainer.train_step(passages, reviews, trainer.train_config)
            model, scheduler, opt = trainer.after_train_step(model, scheduler, opt)

            back_time = timer.hit()
            back_time = (
                back_time / torch.distributed.get_world_size()
                if USE_DEEPSPEED
                else back_time
            )

            # Logging (in terminal and on WANDB)
            timer.hit()
            batch_outputs["Time/Batch"] = back_time

            if trainer.train_config.do_log:
                fn_rank_0(wandb.log, batch_outputs, commit=False)
            if iteration % trainer.train_config.log_interval == 0:
                print_rank_0(
                    f'EPOCH [{epoch}/{trainer.train_config.epochs}]\nBatch Loss: {batch_outputs["Loss/Train"].item()}'
                )
                if trainer.train_config.do_log:
                    fn_rank_0(wandb.log, batch_outputs, commit=True)
            # Checkpoint model and scheduler
            if iteration % trainer.train_config.checkpoint_interval == 0:
                save_iter = (
                    iteration % (20 * trainer.train_config.checkpoint_interval) == 0
                )
                model, scheduler, opt = trainer.before_save(model, scheduler, opt)
                fn_rank_0(
                    save_checkpoint, save_fn, scheduler, opt, iteration, save_iter
                )
                model, scheduler, opt = trainer.after_save(model, scheduler, opt)
            # Run on eval set
            if iteration % trainer.train_config.validate_interval == 0:
                print_rank_0("VALIDATING...")
                model.eval()
                model, scheduler, opt = trainer.before_validate_step(
                    model, scheduler, opt
                )
                eval_data = trainer.construct_dataloader(evalset, tokenizer, multi_gpus)

                eval_out = trainer.eval_step(eval_data)
                model, scheduler, opt = trainer.after_validate_step(
                    model, scheduler, opt
                )

                if eval_out["Loss/Validation"] < best_val:
                    best_val = eval_out["Loss/Validation"]
                    print_rank_0("NEW BEST VALIDATION. SAVING.")
                    print_rank_0(f"Validation Avg Loss: {eval_out['Loss/Validation']}")
                    print_rank_0(
                        f"Validation Avg Accuracy: {eval_out['Acc/Validation']}"
                    )
                    Path(f"./best/{iteration}/").mkdir(parents=True, exist_ok=True)
                    fn_rank_0(save_fn, f"./best/{iteration}/")
                if trainer.train_config.do_log:
                    fn_rank_0(wandb.log, eval_out)
                model.train()

            iteration += 1

            scheduler.step()

            if USE_DEEPSPEED:
                model.module.clamp()
            else:
                model.clamp()


def param_count(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    parser = get_arguments()
    args, _ = parser.parse_known_args()

    # if we are not trying to print an available configuration, continue with training
    if not print_available_configs(args):
        config = CARPConfig.load_yaml(args.config_path)
        train_config = config.train_job
        trainer = get_trainer(train_config.trainer)(train_config)

        sanity_check(args, config)
        args.deepspeed_config = parse_deepspeed_config(
            args,
            trainer.train_config,
            lr=trainer.train_config.learning_rate_init,
            weight_decay=0,
        )

        multi_gpus = (
            args.deepspeed_config is not None
            and "zero_optimization" in args.deepspeed_config
            and args.deepspeed_config["zero_optimization"]["stage"] > 0
        )

        if multi_gpus is True:
            init_process_group(backend="nccl")
            torch.cuda.set_device(torch.distributed.get_rank())

        model = get_model(config, args.load_checkpoint, args.type, args.ckpt_path)
        print_rank_0("N Parameters: " + str(param_count(model)))

        # Logging stuff
        if train_config.do_log:
            fn_rank_0(
                wandb.init,
                name=args.wandb_run_name,
                resume=False,
                config=config.to_dict(),
            )

            # wandb.config.update({"seed": args.seed})
            fn_rank_0(wandb.watch, model)
        dataset, evalset = get_datasets(train_config, args.data_path, args.seed)
        train(model, dataset, evalset, trainer, args, multi_gpus)
