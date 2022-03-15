import torch
import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN

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
P = torch.load("passage_encs.pt")
R = torch.load("review_encs.pt")

P = remove_zeros(P)
R = remove_zeros(R)

# Adjust below when running
REDUCE_DIM = 50
# Z = P # just look at passsages
Z = R # just look at reviews
# Z = torch.cat([P, R]) # look at both together in same domain

# parameters used from UMAP HDBSCAN clustering tutorial
embeddings = UMAP(
    n_neighbors = 30,
    min_dist = 0.0,
    n_components = REDUCE_DIM
).fit_transform(Z)

labels = HDBSCAN(
    min_samples = 8, # min number of labels we'll take
    min_cluster_size = 250 # min number of samples per label we'll take
).fit_predict(embeddings)

# How useful HDBSCAN results are can be glimpsed from
# proportion of labels that are not -1 (-1 meaning HDBSCAN gave up)
quality = len(list(filter(lambda x: x != -1, labels))) / len(labels)
print(quality)