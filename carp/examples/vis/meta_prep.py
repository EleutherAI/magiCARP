import torch
import matplotlib.pyplot as plt

from carp.pytorch.data import *
import numpy as np

from umap import UMAP

from carp.examples.vis.vis_util import scatter_with_names

from joblib import dump, load

# This file assumes extract_embeddings.py was run beforehand
# It's main purpose is just as a preprocessing step for metalabelling code

# Look at SUBSET * len(Z) sized subset of data instead
# (speeds up interactive visualization and umap)
SUBSET = 0.05

def subset(frac, x, names, fixed = True):
    if fixed: torch.manual_seed(0)
    if frac != 1:
        print("Getting random subset")
        def joint_shuffle(A, B): # A tensor, B list
            assert len(A) == len(B)
            inds = torch.randperm(len(A))
            inds = inds[:int(frac * len(A))]
            return A[inds], [B[i] for i in inds]
        return joint_shuffle(x, names)
    return x, names


# ==== EMBEDDING ====
def get_embeddings():

    # Remove tailing zero rows from tensor
    # assumes all entries in T are l2 normalized
    def remove_zeros(T):
        bound = len(T)
        for i in reversed(range(len(T))):
            if T[i].sum() > 0.25: # i.e. not 0
                bound = i + 1
                break
        print("{} encodings found.".format(bound))
        return T[:bound]

    # Load the encodings (assumes running this after extract_embeddings.py)
    print("Loading encodings...")
    try:
        Z = torch.load("review_encs.pt")
    except:
        print("Couldn't find review_encs.pt")
        print("Please run carp.examples.vis.extract_embeddings first")
        print("Note: If above script is not run to completion, this script will still function, but results may be different")
        print("Note: Shuffling in above script will result in visualization not matching reviews to embeddings correctly.")
    Z = remove_zeros(Z)
    

    # This script now assumes you have extracted embeddings already and that they are ordered as in dataset
    # For visualization of clusters

    print("Loading Dataset Strings...")
    pipeline = BaseDataPipeline(path="carp/dataset")
    Z_names = pipeline.reviews
    Z_names = Z_names[:len(Z)]

    Z, Z_names = subset(SUBSET, Z, Z_names)

    print("Performing UMAP reduction...")
    tform = UMAP(
        n_neighbors = 30,
        min_dist = 0.0,
        n_components = 2,
        metric = 'cosine',
        random_state = 42,
        low_memory = True
    ).fit(Z)

    embeddings = tform.transform(Z)

    # Dump the umap to a file
    dump(tform, "umap.joblib")

    return embeddings, Z_names

# If we already ran the script once, can use a checkpoint like this to save lots of time
try:
    embeddings = np.load("plot_encs.npy")
    names = []
    with open("plot_names.txt") as f:
        names = f.readlines()
    print("Succesfully loaded previous run")
except:
    print("Starting fresh run")
    embeddings, names = get_embeddings()
    np.save("plot_encs.npy", embeddings)
    f = open("plot_names.txt", 'w')
    for line in names:
        f.write(line + "\n")
    f.close()
    
# ==== CLUSTERING ====

from scipy.stats import mode
from hdbscan import HDBSCAN

clusterer = HDBSCAN(
    min_samples = 10,
    min_cluster_size = 50, # min number of samples per label we'll take
).fit(embeddings)

labels = clusterer.labels_

# modal label corresponds to the big blob
modal_label = mode(labels).mode[0]

# Return list of indices of all elements that are not a given label
def cull(labels, to_cull):
    return np.where(labels != to_cull)[0].tolist()

inds_1 = cull(labels, modal_label) # cull modal data points
inds_2 = cull(labels, -1) # cull unclassified data points
inds = list(set(inds_1) & set(inds_2)) # intersect to cull both

embeddings = embeddings[inds]

names = [names[i] for i in inds]

# Recluster after the cull
clusterer = HDBSCAN(
    min_samples = 10,
    min_cluster_size = 50, # min number of samples per label we'll take
).fit(embeddings)

labels = clusterer.labels_

inds = cull(labels, -1) # cull unclassified
embeddings = embeddings[inds]
names = [names[i] for i in inds]
labels = labels[inds]

c = labels / labels.max()

scatter_with_names(embeddings[:,0], embeddings[:,1], names, c = c)

# Write everything that was left over to a file to use in meta labelling
dump(clusterer, "hdbscan.joblib")
np.save("umap_review_embeddings.npy", embeddings)