import torch
import numpy as np
from torch.nn.functional import normalize
import math

from carp.configs import CARPConfig
from carp.pytorch.model.architectures import *
from carp.pytorch.data import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# This script returns N_SAMPLES encodings of reviews from the dataset along with indices
# for which review in the dataset each encoding corresponds with

# How many samples are desired?
N_SAMPLES = 10000 # set to -1 to do all
LATENT_DIM = 2048

# load review encodings
def load_encs(path):
    # remove trailing zero vectors
    def remove_zeros(T):
        bound = len(T)
        for i in reversed(range(len(T))):
            if T[i].sum() > 0.25: # i.e. not 0
                bound = i + 1
                break
        print("{} encodings found.".format(bound))
        return T[:bound]
    
    print("Loading Encodings...")
    review_encs = torch.load(path)

    return remove_zeros(review_encs)

# Save encodings
def save_encs(encs, path):
    torch.save(encs, path)

def enc_reviews(N_SAMPLES = N_SAMPLES, force_fresh = True, ind_path = "metalabel_data/embedding_inds.pt", enc_path = "metalabel_data/review_encs.pt", random_state = 0):
    """
    Encodes given number of reviews and saves embeddings into a file. Also saves indices of reviews that were encoded (with respect to the dataset)
    Can take a very long time, so saves checkpoints regularly. Automatically tries to load checkpoint by default.

    :param N_SAMPLES: number of reviews to encode
    :type N_SAMPLES: int

    :param force_fresh: if true, prevents loading of any checkpoints (force a fresh run)
    :type force_fresh: bool

    :param ind_path: path to which to save a tensor indexing which reviews were encoded
    :type ind_path: str

    :param enc_path: path to which to save the review embeddings
    :type enc_path: str

    :param random_state: random seed used to sample reviews from the dataset for encoding
    :type random_state: int
    """
    print("Load Model")
    config = CARPConfig.load_yaml("configs/carp_cloob.yml")
    cloob_model = CARPCloob(config.model)
    cloob_model.load("checkpoints/CLOOB_CP/")
    cloob_model = cloob_model.to(device)
    

    N_CTX = 512
    CHUNK_SIZE = 512

    # Save every N chunks
    SAVE_EVERY = 5

    print("Load Dataset...")
    pipeline = BaseDataPipeline(path="carp/dataset")
    reviews = pipeline.reviews[:-1000] # dont take validation set from training
    reviews = [review[-N_CTX:] for review in reviews] # front truncate
    N = len(reviews)
    
    # Load a previous run by loading indices then encodings if that works
    try:
        assert not force_fresh
        inds = torch.load(ind_path)
        assert len(inds) == N_SAMPLES

        review_encs = load_encs(enc_path).half()
        crnt_ind = len(review_encs) # which review in inds are we at?
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
        ind_chunks = inds.chunk(len(inds) // CHUNK_SIZE)
    else:
        ind_chunks = [inds]
    
    tokenize = cloob_model.passage_encoder.call_tokenizer

    # encode a single batch of txt (list of strings) with review encoder
    def encode(txt_batch):
        tok_out = tokenize(txt_batch)
        x = tok_out["input_ids"]
        mask = tok_out["attention_mask"]
        enc_input = BatchElement(x, mask)

        with torch.no_grad():
            encs = cloob_model.encode_reviews(enc_input).hidden
        encs = encs.cpu().half()
        return encs
    
    def save():
        save_encs(review_encs, enc_path)
    
    # iterate through chunks
    while crnt_ind < N_SAMPLES:
        which_chunk = crnt_ind // CHUNK_SIZE
        inds = ind_chunks[which_chunk]

        # Get chunk and encode it
        review_batch = [reviews[i] for i in inds]
        enc_batch = encode(review_batch)

        review_encs = torch.cat([review_encs, enc_batch])
        
        if which_chunk % SAVE_EVERY == 0:
            print("Progress: [{}/{}]".format(crnt_ind, N_SAMPLES))
            save()
        
        crnt_ind += len(inds)
    
    save()

if __name__ == "__main__":
    enc_reviews(force_fresh=False)