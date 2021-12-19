#TODO: Fix this script so that it works with the new API!!

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import math
import wandb
import argparse
import deepspeed
from constants import *
from util import chunk, generate_indices, get_scheduling_func


# Dataset assumed to be list of pairs on memory
def train(model, dataset, evalset, args=None):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_engine, opt, _, scheduler = deepspeed.initialize(args=args, model=model, model_parameters=parameters)

    # Tokenizes string batch using encoder tokenizer
    # Also adds CLS tokens to end
    def tok(string_batch):
        for i, _ in enumerate(string_batch):
            if len(string_batch[i]) > N_CTX:
                string_batch[i] = string_batch[i][-N_CTX:]

        return model.encA.tok(string_batch)

    # From indices into dataset, gets batch in form of:
    # (passage tokens, passage masks, review tokens, review masks)
    def get_batch_tokens(dataset, inds):
        batch = [dataset[ind] for ind in inds]
        pass_batch = [pair[0] for pair in batch]
        rev_batch = [pair[1] for pair in batch]

        pass_tokens = tok(pass_batch)
        rev_tokens = tok(rev_batch)
        pass_masks = pass_tokens['attention_mask']
        rev_masks = rev_tokens['attention_mask']
        pass_tokens = pass_tokens['input_ids']
        rev_tokens = rev_tokens['input_ids']

        return pass_tokens, pass_masks, rev_tokens, rev_masks

    # Get encodings and validates them (gets loss and accuracy) without grad
    def encode_and_val(pass_mbs, rev_mbs):
        with torch.no_grad():
            pass_encs = [model_engine.module.encodeX(tokens, masks, device=model_engine.device)
                for (tokens, masks) in pass_mbs]
            
            rev_encs = [model_engine.module.encodeY(tokens, masks, device=model_engine.device)
                for (tokens, masks) in rev_mbs]
        
            test_loss, test_acc = model_engine.module.cLoss(torch.cat(pass_encs), torch.cat(rev_encs), device=model_engine.device)
        return pass_encs, rev_encs, test_loss, test_acc

    #scheduler = LambdaLR(opt, get_scheduling_func())
    if LOAD_CHECKPOINT:
        opt.load_state_dict(torch.load("./opt.pt"))
    
    model_engine.train() 
    
    dataset_size = len(dataset)
    evalset_size = len(evalset)

    iteration = 0

    for epoch in range(EPOCHS):
        batches_inds = generate_indices(dataset_size, BATCH_SIZE)
        for batch_inds in batches_inds:
            pass_tokens, pass_masks, rev_tokens, rev_masks = get_batch_tokens(dataset, batch_inds)
            microbatch_inds = generate_indices(len(batch_inds), MICROBATCH_SIZE, shuffle = False)

            # Split tokens and masks into these microbatches
            pass_mbs = [(pass_tokens[ind], pass_masks[ind]) for ind in microbatch_inds]
            rev_mbs = [(rev_tokens[ind], rev_masks[ind]) for ind in microbatch_inds]

            # Initially get all encodings without grad
            pass_encs, rev_encs, forward_loss, forward_acc = encode_and_val(pass_mbs, rev_mbs)

            #opt.zero_grad()
            # Encode passages in microbatches (with grad)
            for index, (tokens, masks) in enumerate(pass_mbs):
                pass_tmp = pass_encs.copy()
                pass_tmp[index] = model_engine.module.encodeX(tokens, masks, device=model_engine.device)
                loss, _ = model_engine.module.cLoss(torch.cat(pass_tmp), torch.cat(rev_encs), device=model_engine.device)
                model_engine.backward(loss)

            # Encode reviews in microbatches (with grad)
            for index, (tokens, masks) in enumerate(rev_mbs):
                rev_tmp = rev_encs.copy()
                rev_tmp[index] = model_engine.module.encodeY(tokens, masks, device=model_engine.device)
                loss, _ = model_engine.module.cLoss(torch.cat(pass_encs), torch.cat(rev_tmp), device=model_engine.device)
                model_engine.backward(loss)

            model_engine.step()
            scheduler.step()


            # Logging (in terminal and on WANDB)
            if iteration % LOG_INTERVAL == 0:
                print("EPOCH [" + str(epoch) + "/" + str(EPOCHS) +
                  "] Batch Loss: " + str(forward_loss.item()))
                if DO_LOG:
                    wandb.log({"Loss/train": forward_loss,
                            "Acc/train": forward_acc})
            # Checkpoint model and scheduler
            if iteration % CHECKPOINT_INTERVAL == 0:
                print("SAVING...")
                # Only save extra once every 20
                if iteration % (20 * CHECKPOINT_INTERVAL) == 0:
                    torch.save(model_engine.module.state_dict(), "./checkpoints/" + str(iteration) \
                           + "params.pt")
                torch.save(model_engine.module.state_dict(), "./params.pt")
                #torch.save(opt.state_dict(), "./opt.pt")
            # Run on eval set
            if (iteration+1) % VALIDATE_INTERVAL == 0:
                print("VALIDATING...")
                model_engine.eval()
                
                pass_t, pass_m, rev_t, rev_m = get_batch_tokens(evalset, np.arange(evalset_size))
                microbatch_inds = generate_indices(evalset_size, MICROBATCH_SIZE, shuffle = False)

                pass_mbs = [(pass_t[ind], pass_m[ind]) for ind in microbatch_inds]
                rev_mbs = [(rev_t[ind], rev_m[ind]) for ind in microbatch_inds]
                    
                _, _, val_loss, val_acc = encode_and_val(pass_mbs, rev_mbs)
                val_loss = val_loss.item()
                val_acc = val_acc.item()

                print("Validation Avg Loss: " + str(val_loss))
                print("Validation Avg Accuracy: " + str(val_acc))
                if DO_LOG:
                    wandb.log({"Loss/validation": val_loss})
                    wandb.log({"Acc/validation": val_acc})
                model_engine.train()
            
            iteration += 1
            model_engine.module.clamp()

from model import ContrastiveModel
from encoder import TextEncoder
from dataloading import get_dataset
import util

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='CARP')
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    args=parser.parse_args()
    model = ContrastiveModel(TextEncoder(), TextEncoder())
    if LOAD_CHECKPOINT: model.load_state_dict(torch.load("./params.pt"))

    


    # Logging stuff
    if DO_LOG:
        wandb.init(project = "CARP", entity = "EleutherAI", resume = LOAD_CHECKPOINT)
        wandb.watch(model)
    
    dataset, evalset = get_dataset()
    print("data loaded")

    train(model, dataset, evalset, args)
