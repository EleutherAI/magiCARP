from argparse import ArgumentParser
from carp.clock import Clock
from carp.configs import CARPConfig, TrainConfig

from carp.pytorch.model.architectures import get_architecture, get_architecture_names
from carp.pytorch.model.encoders import get_encoder_names
from carp.util import get_scheduling_func
from carp.pytorch.dataset import CarpDataset, tokenizer_factory

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split, RandomSampler, Subset
import wandb


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=False)
    parser.add_argument("--ckpt_path", type=str, required=False)
    parser.add_argument("--config_path", type=str, default="./base_config.yml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load_checkpoint", type=bool, default=False)
    parser.add_argument("--wandb_run_name", type=str)
    parser.add_argument("--seed", type=float, default=42)
    parser.add_argument("--type", type=str, default="CARP")
    parser.add_argument("--get_architectures", action='store_true')
    parser.add_argument("--get_encoders", action='store_true')
    return parser


def get_model(config: CARPConfig, load_checkpoint: bool,\
    model_type : str = "CARP", ckpt_path = None):

    model = get_architecture(model_type)(config.model)
    if load_checkpoint:
        if ckpt_path is None:
            model.load_state_dict(torch.load("./params.pt"))
        else:
            model.load_state_dict(torch.load(ckpt_path))
    model.cuda()
    if config.train_job.use_half:
        model.half()
    return model


def get_datasets(config, data_path, random_seed=None):
    carp = CarpDataset(config.dupe_protection, data_path)
    size = len(carp)

    seed = torch.manual_seed(random_seed)
    if config.eval_selection == 'random':
        splits = [size - config.validation_size, config.validation_size]
        return random_split(carp, splits, generator=seed)
    elif config.eval_selection == 'final_n':
        train_indices = list(range(size - config.validation_size))
        eval_indices = list(range(size - config.validation_size, size))
        return Subset(carp, train_indices), Subset(carp, eval_indices)
    else:
        raise NotImplementedError('The only valid options for `eval_selection` are "random" and "final_n"')


def save_checkpoint(model, scheduler, opt, iter: int, save_iter: bool):
    print("SAVING...")
    # Only save extra once every 20
    if save_iter:
        torch.save(
            model.state_dict(),
            f"./checkpoints/{iter}params.pt",
        )
    torch.save(model.state_dict(), "./params.pt")
    torch.save(scheduler.state_dict(), "./schedule.pt")
    torch.save(opt.state_dict(), "./opt.pt")


# Dataset assumed to be list of pairs on memory
def train(model, dataset: CarpDataset, evalset: CarpDataset, config: TrainConfig, args):
    # Tokenizes string batch using encoder tokenizer
    LEARNING_RATE_INIT = config.learning_rate_init
    LOAD_CHECKPOINT = args.load_checkpoint
    tokenizer = tokenizer_factory(model.passage_encoder.tok, config.n_ctx)
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
                save_checkpoint(model, scheduler, opt, iteration, save_iter)
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

    if args.get_architectures:
        print("FORMAT: Architecture") 
        print("Available architectures are:")
        print("***************")
        print("\n".join(get_architecture_names()))
    elif args.get_encoders:
        print("FORMAT: Encoder")
        print("Available encoders are:")
        print("***************")
        print("\n".join(get_encoder_names()))
    else:
        config = CARPConfig.load_yaml(args.config_path)
        train_config = config.train_job
        model = get_model(config, args.load_checkpoint, args.type, args.ckpt_path)
        print("N Parameters: " + str(param_count(model)))
        # Logging stuff
        if train_config.do_log:
            wandb.init(
                name=args.wandb_run_name,
                resume=False,
                config=config.to_dict(),
            )
            wandb.config.update({'seed': args.seed})
            wandb.watch(model)
        dataset, evalset = get_datasets(train_config, args.data_path, args.seed)
        train(model, dataset, evalset, train_config, args)
