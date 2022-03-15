import torch
import matplotlib.pyplot as plt

from carp.pytorch.data import *

from umap import UMAP

from carp.examples.vis.vis_util import scatter_with_names

# Look at SUBSET * len(Z) sized subset of data instead
# (speeds up interactive visualization and umap)
SUBSET = 0.2

# Remove tailing zero rows from tensor
# assumes all entries in T are l2 normalized
def remove_zeros(T):
    bound = len(T)
    for i in range(len(T)):
        if (T[i]**2).sum() < 0.25: # i.e. not 1
            bound = i
            break
    return T[:bound]

# Load the encodings (assumes running this after extract_embeddings.py)
print("Loading encodings...")
P = torch.load("passage_encs.pt")
R = torch.load("review_encs.pt")

P = remove_zeros(P)
R = remove_zeros(R)

# Adjust below when running
REDUCE_DIM = 50
# Z = P # just look at passsages
Z = R # just look at reviews
# Z = torch.cat([P, R]) # look at both together in same domain

# This script now assumes you have extracted embeddings already and that they are ordered as in dataset
# For visualization of clusters

print("Loading Dataset Strings...")
pipeline = BaseDataPipeline(path="carp/dataset")
P_names = pipeline.passages
R_names = pipeline.reviews

P_names = P_names[:len(P)]
R_names = R_names[:len(R)]

# Z_names = P_names
Z_names = R_names
# Z_names = P_names + R_names

if SUBSET != 1:
    print("Getting random subset")
    def joint_shuffle(A, B): # A tensor, B list
        assert len(A) == len(B)
        inds = torch.randperm(len(A))
        inds = inds[:int(SUBSET * len(Z))]
        return A[inds], [B[i] for i in inds]
    Z, Z_names = joint_shuffle(Z, Z_names)

print("Performing UMAP reduction...")
embeddings = UMAP(
    n_neighbors = 30,
    min_dist = 0.0,
    n_components = 2,
    metric = 'cosine'
).fit_transform(Z)
# get 2d umap for visualization

print("Plotting...")
sc_x = embeddings[:,0]
sc_y = embeddings[:,1]
scatter_with_names(sc_x, sc_y, Z_names)
