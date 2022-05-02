import math
import sys
from typing import Iterable

import numpy as np
import torch
from torch.nn.functional import normalize

from carp.configs import CARPConfig
from carp.examples.encodings.util import chunk, load_encs, save_encs
from carp.pytorch.data import *
from carp.pytorch.model.architectures import BaseModel
from carp.pytorch.model.architectures.carp import CARP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def enc_reviews(
    N_SAMPLES: int,
    force_fresh: bool,
    CHUNK_SIZE: int,
    SAVE_EVERY: int,
    model: BaseModel,
    txt_data: Iterable[str],
    N_CTX: int = 512,
    ind_path: str = "carp/examples/encodings/rev_embedding_inds.pt",
    enc_path: str = "carp/examples/encodings/review_encs.pt",
    random_state: int = 0,
):
    """
    Encodes given number of reviews and saves embeddings into a file. Also saves indices of reviews that were encoded (with respect to the dataset)
    Can take a very long time, so saves checkpoints regularly. Automatically tries to load checkpoint by default.

    :param N_SAMPLES: number of reviews to encode
    :type N_SAMPLES: int

    :param force_fresh: if true, prevents loading of any checkpoints (force a fresh run)
    :type force_fresh: bool

    :param CHUNK_SIZE: number of reviews to encode at once
    :type CHUNK_SIZE: int

    :param SAVE_EVERY: number of reviews to encode before saving checkpoint
    :type SAVE_EVERY: int

    :param model: model to use for encoding
    :type model: carp.pytorch.model.architectures.BaseModel

    :param txt_data: iterable of reviews to encode
    :type txt_data: Iterable[str]

    :param N_CTX: context size for encoding (specific to model being used)
    :type N_CTX: int

    :param ind_path: path to which to save a tensor indexing which reviews were encoded
    :type ind_path: str

    :param enc_path: path to which to save the review embeddings
    :type enc_path: str

    :param random_state: random seed used to sample reviews from the dataset for encoding
    :type random_state: int
    """
    N_CTX = 512
    txt_data = [txt[-N_CTX:] for txt in txt_data]
    LATENT_DIM = model.latent_dim

    N = len(txt_data)

    # Load a previous run by loading indices then encodings if that works
    try:
        assert not force_fresh
        inds = torch.load(ind_path)
        assert len(inds) == N_SAMPLES

        review_encs = load_encs(enc_path).half()
        crnt_ind = len(review_encs)  # which review in inds are we at?
    except:
        # generate indices of the reviews we will use
        torch.manual_seed(random_state)
        inds = torch.randperm(N)[:N_SAMPLES]
        torch.save(inds, ind_path)

        review_encs = torch.zeros(0, LATENT_DIM).half()
        crnt_ind = 0

    # Chunk the indices based on chunk size
    n_chunks = len(inds) // CHUNK_SIZE
    if n_chunks > 0:
        ind_chunks = chunk(inds, CHUNK_SIZE)
    else:
        ind_chunks = [inds]

    tokenize = model.review_encoder.call_tokenizer

    # encode a single batch of txt (list of strings) with review encoder
    def encode(txt_batch):
        tok_out = tokenize(txt_batch)
        x = tok_out["input_ids"].to(device)
        mask = tok_out["attention_mask"].to(device)
        enc_input = BatchElement(x, mask)

        with torch.no_grad():
            encs = model.encode_reviews(enc_input).hidden
        encs = encs.cpu().half()
        return encs

    # save the encodings so far
    def save():
        save_encs(review_encs, enc_path)

    # iterate through chunks
    while crnt_ind < N_SAMPLES:
        which_chunk = crnt_ind // CHUNK_SIZE
        inds = ind_chunks[which_chunk]

        # Get chunk and encode it
        review_batch = [txt_data[i] for i in inds]
        enc_batch = encode(review_batch)

        review_encs = torch.cat([review_encs, enc_batch])

        if which_chunk % SAVE_EVERY == 0:
            print("Progress: [{}/{}]".format(crnt_ind, N_SAMPLES))
            save()

        crnt_ind += len(inds)

    save()


if __name__ == "__main__":
    # Decide whether or not to do fresh run from command line
    if len(sys.argv) >= 2:
        if sys.argv[1] == "FRESH" or sys.argv[1] == "fresh":
            force_fresh = True
        else:
            force_fresh = False

    # Use CARP Large by default
    print("Load Model...")
    config = CARPConfig.load_yaml("configs/carp_l.yml")
    model = CARP(config.model)
    model.load("checkpoints/CARP_L/")
    model = model.to(device)

    # And the Story-Critique dataset
    pipeline = BaseDataPipeline(path="carp/dataset")

    enc_reviews(
        N_SAMPLES=10000,
        force_fresh=force_fresh,
        CHUNK_SIZE=256,
        SAVE_EVERY=5,
        model=model,
        txt_data=pipeline.reviews,
        N_CTX=512,
        random_state=0,
    )
