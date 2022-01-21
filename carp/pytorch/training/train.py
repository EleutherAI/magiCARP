from argparse import ArgumentParser
from carp.clock import Clock
from carp.configs import CARPConfig, TrainConfig
from carp.pytorch.model.architectures import get_architecture
from carp.pytorch.training import get_orchestrator
from carp.util import get_scheduling_func
from carp.pytorch.data import get_datapipeline, BaseDataPipeline
from carp.pytorch.training import get_orchestrator, BaseOrchestrator
from carp.pytorch.training.utils import print_available_configs

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split, RandomSampler, Subset
import wandb
from pathlib import Path


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=False)
    parser.add_argument("--ckpt_path", type=str, required=False)
    parser.add_argument("--config_path", type=str, default="./base_config.yml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load_checkpoint", action='store_true')
    parser.add_argument("--wandb_run_name", type=str)
    parser.add_argument("--seed", type=float, default=42)
    parser.add_argument("--type", type=str, default="CARP")
    parser.add_argument("--get_architectures", action='store_true')
    parser.add_argument("--get_encoders", action='store_true')
    parser.add_argument("--get_datapipelines", action='store_true')
    parser.add_argument("--get_orchestrators", action='store_true')
    return parser


def get_model(config: CARPConfig, load_checkpoint: bool,\
    model_type : str = "CARP", ckpt_path = None):

    model = get_architecture(model_type)(config.model)
    if load_checkpoint:
        if ckpt_path is None:
            model.load("./output/")
        else:
            model.load(ckpt_path)
        print("Checkpoint loaded!")
    model.cuda()
    if config.train_job.use_half:
        model.half()
    return model


def get_datasets(config, data_path, random_seed=None):
    dataset = get_datapipeline(config.data_pipeline)(config.dupe_protection, data_path)
    size = len(dataset)

    seed = torch.manual_seed(random_seed)
    if config.eval_selection == 'random':
        splits = [size - config.validation_size, config.validation_size]
        return random_split(dataset, splits, generator=seed)
    elif config.eval_selection == 'final_n':
        train_indices = list(range(size - config.validation_size))
        eval_indices = list(range(size - config.validation_size, size))
        return Subset(dataset, train_indices), Subset(dataset, eval_indices)
    else:
        raise NotImplementedError('The only valid options for `eval_selection` are "random" and "final_n"')


def save_checkpoint(model, scheduler, opt, iter: int, save_iter: bool):
    print("SAVING...")
    # Only save extra once every 20
    if save_iter:
        Path(f"./checkpoints/{iter}/").mkdir(parents=True, exist_ok=True)
        model.save(f"./checkpoints/{iter}/")
        
    Path("./output/").mkdir(parents=True, exist_ok=True)
    model.save("./output/")
    torch.save(scheduler.state_dict(), "./output/schedule.pt")
    torch.save(opt.state_dict(), "./output/opt.pt")


# Dataset assumed to be list of pairs on memory
def train(model,
    dataset: BaseDataPipeline,
    evalset: BaseDataPipeline,
    orchestrator : BaseOrchestrator,
    args):
    # Tokenizes string batch using encoder tokenizer
    LEARNING_RATE_INIT = orchestrator.train_config.learning_rate_init
    LOAD_CHECKPOINT = args.load_checkpoint

    # setup data pipeline. model is needed 
    tokenizer = orchestrator.construct_tokenizer(model.passage_encoder)

    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE_INIT, weight_decay=0,eps = 1e-5)
    scheduler = LambdaLR(opt, get_scheduling_func(orchestrator.train_config))
    scaler = torch.cuda.amp.GradScaler()
    if LOAD_CHECKPOINT:
        try:
            if args.ckpt_path is None:
                scheduler.load_state_dict(torch.load("./output/schedule.pt"))
                opt.load_state_dict(torch.load("./output/opt.pt"))
            else:
                scheduler.load_state_dict(torch.load(args.ckpt_path+"schedule.pt"))
                opt.load_state_dict(torch.load(args.ckpt_path+"opt.pt"))
        except:
            print("Unable to load scheduler and/or optimizer. Continuing.")
    model.train()
    iteration = 0
    timer = Clock()
    for epoch in range(orchestrator.train_config.epochs):
        model, scheduler, opt = orchestrator.on_epoch_start(model, scheduler, opt)
        train_data = orchestrator.construct_dataloader(dataset, tokenizer)

        for passages, reviews in train_data:
            timer.hit()
            model, scheduler, opt = orchestrator.before_train_step(model, scheduler, opt)
            batch_outputs = model.train_step(passages, reviews, orchestrator.train_config, opt, scaler)
            model, scheduler, opt = orchestrator.after_train_step(model, scheduler, opt)

            back_time = timer.hit()
            # Logging (in terminal and on WANDB)
            timer.hit()
            batch_outputs["Time/Batch"] = back_time
            if orchestrator.train_config.do_log:
                wandb.log(batch_outputs, commit=False)
            if iteration % orchestrator.train_config.log_interval == 0:
                print(
                    f'EPOCH [{epoch}/{orchestrator.train_config.epochs}]\nBatch Loss: {batch_outputs["Loss/Train"].item()}'
                )
                if orchestrator.train_config.do_log:
                    wandb.log(batch_outputs, commit=True)
            # Checkpoint model and scheduler
            if iteration % orchestrator.train_config.checkpoint_interval == 0:
                save_iter = iteration % (20 * orchestrator.train_config.checkpoint_interval) == 0
                model, scheduler, opt = orchestrator.before_save(model, scheduler, opt)
                save_checkpoint(model, scheduler, opt, iteration, save_iter)
                model, scheduler, opt = orchestrator.after_save(model, scheduler, opt)
            # Run on eval set
            if iteration % orchestrator.train_config.validate_interval == 0:
                print("VALIDATING...")
                model.eval()
                model, scheduler, opt = orchestrator.before_validate_step(model, scheduler, opt)
                eval_data = orchestrator.construct_dataloader(evalset, tokenizer)

                eval_out = model.eval_step(eval_data)
                model, scheduler, opt = orchestrator.after_validate_step(model, scheduler, opt)
                print(f"Validation Avg Loss: {eval_out['Loss/Validation']}")
                print(f"Validation Avg Accuracy: {eval_out['Acc/Validation']}")
                if orchestrator.train_config.do_log:
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
    # if we are not trying to print an available configuration, continue with training
    if not print_available_configs(args):
        config = CARPConfig.load_yaml(args.config_path)
        train_config = config.train_job
        model = get_model(config, args.load_checkpoint, args.type, args.ckpt_path)
        orchestrator = get_orchestrator(train_config.orchestrator)(train_config)
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
        train(model, dataset, evalset, orchestrator, args)
