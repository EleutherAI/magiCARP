import torch
import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN

from carp.pytorch.data import *
from carp.examples.vis.vis_util import scatter_with_names

SUBSET = 0.3
REDUCE_DIM = 50

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

# Z = P # just look at passsages
Z = R # just look at reviews
# Z = torch.cat([P, R]) # look at both together in same domain

# Load strings from dataset
print("Loading Dataset Strings...")
pipeline = BaseDataPipeline(path="carp/dataset")
P_names = pipeline.passages
R_names = pipeline.reviews

P_names = P_names[:len(P)]
R_names = R_names[:len(R)]

# Z_names = P_names
Z_names = R_names
# Z_names = P_names + R_names

# Get random subset of data
if SUBSET != 1:
    print("Getting random subset...")
    def joint_shuffle(A, B): # A tensor, B list
        assert len(A) == len(B)
        inds = torch.randperm(len(A))
        inds = inds[:int(SUBSET * len(Z))]
        return A[inds], [B[i] for i in inds]
    Z, Z_names = joint_shuffle(Z, Z_names)

# parameters used from UMAP HDBSCAN clustering tutorial
print("Performing 1st UMAP reduction...")
embeddings = UMAP(
    n_neighbors = 30,
    min_dist = 0.0,
    n_components = REDUCE_DIM,
    metric = 'cosine'
).fit_transform(Z)

print("Performing HDBSCAN Clustering...")
labels = HDBSCAN(
    min_samples = 8, # min number of labels we'll take
    min_cluster_size = 250, # min number of samples per label we'll take
).fit_predict(embeddings)

# How useful HDBSCAN results are can be glimpsed from
# proportion of labels that are not -1 (-1 meaning HDBSCAN gave up)
quality = len(list(filter(lambda x: x != -1, labels))) / len(labels)
print(quality)

#print("Performing 2nd UMAP reduction...")
#embeddings = UMAP(
#    n_neighbors = 30,
#    min_dist = 0.0,
#    n_components = 2,
#    metric = 'cosine'
#).fit_transform(Z)
# get 2d umap for visualization

print("Plotting...")
sc_x = embeddings[:,0]
sc_y = embeddings[:,1]
scatter_with_names(sc_x, sc_y, Z_names, labels)
