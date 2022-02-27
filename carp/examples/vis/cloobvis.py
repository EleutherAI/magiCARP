import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import normalize

from carp.configs import CARPConfig
from carp.pytorch.model.architectures import *
from carp.pytorch.data import *

from vis_util import spherical_coord, scatter_with_names


if __name__ == "__main__":
    # Load model
    config = CARPConfig.load_yaml("configs/carp_cloob.yml")
    cloob_model = CARPCloob(config.model)
    cloob_model.load("checkpoints/CLOOB_CP/")
    cloob_model = cloob_model.cuda()

    # Extract raw reviews and passages from pipeline
    pipeline = BaseDataPipeline(path="carp/dataset")
    passages = pipeline.passages
    reviews = pipeline.reviews
    N = len(passages)
    print(N)

# Sample random batch from dataset
def get_random_batch(size):
    inds = torch.randint(0, N, (size,))
    p = [passages[ind][:512] for ind in inds]
    r = [reviews[ind][:512] for ind in inds]

    return p, r

# Use tokenizer on a batch of passages and reviews
def tokenizer_batch(p,  r):
    p_tok = cloob_model.passage_encoder.call_tokenizer(p)
    r_tok = cloob_model.passage_encoder.call_tokenizer(p)
    return p_tok, r_tok
    
# construct batch element from a tokenizer output
def make_batch_elem(tok_out):
    return BatchElement(tok_out['input_ids'],
                       tok_out['attention_mask'])

# Sample random batch from dataset and encode it
def encode_rand_batch(size):
    p_batch_s, r_batch_s = get_random_batch(size)
    p_tok, r_tok = tokenizer_batch(p_batch_s, r_batch_s)
    p_batch = make_batch_elem(p_tok)
    r_batch = make_batch_elem(r_tok)

    with torch.no_grad():
        pass_encs, rev_encs = \
                cloob_model.calculate_embeddings([p_batch], [r_batch])
    # -> [[pass_encs], [rev_encs]]
    pass_encs = pass_encs[0].float()
    rev_encs = rev_encs[0].float()
    
    pass_encs = normalize(pass_encs)
    rev_encs = normalize(rev_encs)

    return (pass_encs, p_batch_s), \
        (rev_encs, r_batch_s)

# Given encodings and batch (strings), return PCA result
# and batch sorted by first value in PCA vectors
def pca_sort(encs, batch):
    U,_,_ = torch.pca_lowrank(encs, q = 2)
    U = U.detach().cpu().numpy()

    # Sort by x value for easier visualization
    U = [[u, pass_] for (u, pass_) in zip(U, batch)]
    U = sorted(U, key = lambda x: x[0][0])
    batch = [x[1] for x in U]

    # Get 2d vectors from PCA
    U = [x[0] for x in U]

    return U, batch

# Mode = "R" or "P" review or passage
def encode_and_draw(size, mode = "R"):
    if mode == "R":
        _, (encs, batch) = encode_rand_batch(size)
    else:
        (encs, batch), _ = encode_rand_batch(size)
    
    U, batch = pca_sort(encs, batch)
    U_x = np.array([u[0] for u in U])
    U_y = np.array([u[1] for u in U])
    
    scatter_with_names(U_x, U_y, batch)

# Mode = "R" or "P" review or passage
def spherecoord_ead(size, mode = "R"):
    if mode == "R":
        _, (encs, batch) = encode_rand_batch(size)
    else:
        (encs, batch), _ = encode_rand_batch(size)
    encs = spherical_coord(encs)
    
    U, batch = pca_sort(encs, batch)
    U_x = np.array([u[0] for u in U])
    U_y = np.array([u[1] for u in U])
    
    scatter_with_names(U_x, U_y, batch)

# Compare sphere to normal
def spherebasecompare(size, mode = "R"):
    if mode == "R":
        _, (encs, batch) = encode_rand_batch(size)
    else:
        (encs, batch), _ = encode_rand_batch(size)
    
    sphere_encs = spherical_coord(encs)

    U_sphere, batch_sphere = pca_sort(sphere_encs, batch)
    U, batch = pca_sort(encs, batch)

    U_x = np.array([u[0] for u in U])
    U_y = np.array([u[1] for u in U])
    
    scatter_with_names(U_x, U_y, batch)

    U_x = np.array([u[0] for u in U_sphere])
    U_y = np.array([u[1] for u in U_sphere])

    scatter_with_names(U_x, U_y, batch_sphere)

# Main functions of interest here are:
# encode_and_draw(size, mode) which makes scatter plot of PCA embeddings
# spherecoord_ead(size, mode) which does the same but with spherical coordinates of embeddings
# spherebasecompare(size, mode) does both one after the other for direct comparison
if __name__ == "__main__":
    #spherebasecompare(1024, mode = "R")
    spherecoord_ead(512)