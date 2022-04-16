from argparse import ArgumentParser

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, Subset, random_split

import wandb
from carp.clock import Clock
from carp.configs import CARPConfig, TrainConfig
from carp.pytorch.data import BaseDataPipeline
from carp.pytorch.model import CARPMomentum
from carp.util import get_scheduling_func


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--linear_projection", type=bool)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--lr_ramp_steps", type=int)
    parser.add_argument("--lr_decay_steps", type=int)
    parser.add_argument("--learning_rate_target", type=float)
    parser.add_argument("--proj_dropout", type=float)
    parser.add_argument("--use_half", type=bool)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--n_ctx", type=int)
    parser.add_argument("--latent_dim", type=int)
    parser.add_argument("--data_path", type=str, default="carp/dataset")
    parser.add_argument("--config_path", type=str, default="./sentence_conf.yml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load_checkpoint", type=bool, default=False)
    parser.add_argument("--seed", type=float, default=3.1415926)
    return parser


def get_model(config: CARPConfig, load_checkpoint: bool):
    model = CARPMomentum(config.model)
    if load_checkpoint:
        model.load_state_dict(torch.load("./params.pt"))
    model.cuda()
    if config.train_job.use_half:
        pass
    return model


def get_datasets(config, data_path, random_seed):
    carp = BaseDataPipeline(config.dupe_protection, data_path)
    size = len(carp)
    seed = torch.manual_seed(random_seed)
    if config.eval_selection == "random":
        splits = [size - config.validation_size, config.validation_size]
        return random_split(carp, splits, generator=seed)
    elif config.eval_selection == "final_n":
        train_indices = list(range(size - config.validation_size))
        eval_indices = list(range(size - config.validation_size, size))
        return Subset(carp, train_indices), Subset(carp, eval_indices)
    else:
        raise NotImplementedError(
            'The only valid options for `eval_selection` are "random" and "final_n"'
        )


def save_checkpoint(model, scheduler, opt, iter: int, save_iter: bool, save_folder: str):
    print("SAVING...")
    # Only save extra once every 20
    if save_iter:
        torch.save(
            model.state_dict(),
            f"./checkpoints/{save_folder}/{iter}params.pt",
        )
    torch.save(model.state_dict(), f"./{save_folder}/params.pt")
    torch.save(scheduler.state_dict(), f"./{save_folder}/schedule.pt")
    torch.save(opt.state_dict(), f"./{save_folder}/opt.pt")


# Dataset assumed to be list of pairs on memory
def train(
    model,
    dataset: BaseDataPipeline,
    evalset: BaseDataPipeline,
    config: TrainConfig,
    args,
):
    # Tokenizes string batch using encoder tokenizer
    LEARNING_RATE_INIT = config.learning_rate_init
    LOAD_CHECKPOINT = args.load_checkpoint
    tokenizer = BaseDataPipeline.tokenizer_factory(
        model.passage_encoder.tok, config.n_ctx
    )
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE_INIT, weight_decay=0)
    scheduler = LambdaLR(opt, get_scheduling_func(config))
    scaler = torch.cuda.amp.GradScaler()
    if LOAD_CHECKPOINT:
        scheduler.load_state_dict(torch.load("./schedule.pt"))
        opt.load_state_dict(torch.load("./opt.pt"))
    model.train()
    iteration = 0
    timer = Clock()
    for epoch in range(config.epochs):
        train_sampler = RandomSampler(dataset)
        train_data = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=3,
            collate_fn=tokenizer,
            pin_memory=True,
        )
        for passages, reviews in train_data:
            timer.hit()
            batch_outputs = model.train_step(passages, reviews, config, opt, scaler)
            back_time = timer.hit()
            # Logging (in terminal and on WANDB)
            timer.hit()
            batch_outputs["Time/Batch"] = back_time
            if config.do_log:
                wandb.log(batch_outputs, commit=False)
            if iteration % config.log_interval == 0:
                print(
                    f'EPOCH [{epoch}/{config.epochs}]\nBatch Loss: {batch_outputs["Loss/Train"].item()}'
                )
                if config.do_log:
                    wandb.log(batch_outputs, commit=True)
            # Checkpoint model and scheduler
            if iteration % config.checkpoint_interval == 0:
                save_iter = iteration % (20 * config.checkpoint_interval) == 0
                save_checkpoint(model, scheduler, opt, iteration, save_iter, config.save_folder)
            # Run on eval set
            if iteration % config.validate_interval == 0:
                print("VALIDATING...")
                model.eval()
                eval_sampler = RandomSampler(evalset)
                eval_data = DataLoader(
                    evalset,
                    batch_size=config.microbatch_size,
                    sampler=eval_sampler,
                    collate_fn=tokenizer,
                )
                eval_out = model.eval_step(eval_data)
                print(f"Validation Avg Loss: {eval_out['Loss/Validation']}")
                print(f"Validation Avg Accuracy: {eval_out['Acc/Validation']}")
                if config.do_log:
                    wandb.log(eval_out)
                model.train()
            iteration += 1
            scheduler.step()
            model.clamp()


def param_count(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    parser = get_arguments()
    args, _ = parser.parse_known_args()
    config = CARPConfig.load_yaml(args.config_path)
    train_config = config.train_job
    config.train_job.lr_ramp_steps = args.lr_ramp_steps
    config.train_job.lr_decay_steps = args.lr_decay_steps
    config.train_job.learning_rate_target = args.learning_rate_target
    config.train_job.n_ctx = args.n_ctx
    config.train_job.epochs = args.epochs
    config.train_job.use_half = args.use_half
    config.model.latent_dim = args.latent_dim
    config.model.momentum = args.momentum
    config.model.linear_projection = args.linear_projection
    config.model.proj_dropout = args.proj_dropout
    model = get_model(config, args.load_checkpoint)
    print("N Parameters: " + str(param_count(model)))

    # Logging stuff
    if train_config.do_log:
        wandb.init(
            resume=False,
            config=config.to_dict(),
        )
        wandb.config.update({"seed": args.seed})
        wandb.watch(model)
        print("wandb configured")
    dataset, evalset = get_datasets(train_config, args.data_path, args.seed)
    train(model, dataset, evalset, train_config, args)
    wandb.finish()
