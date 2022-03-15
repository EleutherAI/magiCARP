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

# parameters used from UMAP HDBSCAN clustering tutorial
print("Performing 1st UMAP reduction...")
embeddings = UMAP(
    n_neighbors = 30,
    min_dist = 0.0,
    n_components = REDUCE_DIM
).fit_transform(Z)

print("Performing HDBSCAN Clustering...")
labels = HDBSCAN(
    min_samples = 8, # min number of labels we'll take
    min_cluster_size = 250 # min number of samples per label we'll take
).fit_predict(embeddings)

# How useful HDBSCAN results are can be glimpsed from
# proportion of labels that are not -1 (-1 meaning HDBSCAN gave up)
quality = len(list(filter(lambda x: x != -1, labels))) / len(labels)
print(quality)

# Exit here if you're just checking HDBSCAN quality
#exit()

# For visualization of clusters
from carp.pytorch.data import *

print("Loading Dataset Strings...")
pipeline = BaseDataPipeline(path="carp/dataset")
P_names = pipeline.passages
R_names = pipeline.reviews

P_names = P_names[:len(P)]
R_names = R_names[:len(R)]

# Z_names = P_names
Z_names = R_names
# Z_names = P_names + R_names

from carp.examples.vis.vis_util import scatter_with_names

# data not assigned to a cluster will be black
labels = ['black' if label == -1 else label for label in labels] 
print("Performing 2nd UMAP reduction...")
embeddings = UMAP(
    n_neighbors = 30,
    min_dist = 0.0,
    n_components = 2
).fit_transform(Z)
# get 2d umap for visualization

print("Plotting...")
sc_x = embeddings[:,0]
sc_y = embeddings[:,1]
scatter_with_names(sc_x, sc_y, Z_names, labels)
