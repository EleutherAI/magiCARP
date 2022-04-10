import torch
import numpy as np
import random

from scipy.stats import mode
from hdbscan import HDBSCAN
from umap import UMAP
from joblib import dump, load

from carp.pytorch.data import *

# Load encodings
R = torch.load("metalabel_data/review_encs.pt")
inds = torch.load("metalabel_data/embedding_inds.pt")
reduce_dim = -1 # set to -1 to not reduce dim

# Load dataset
print("Load Dataset...")
pipeline = BaseDataPipeline(path="carp/dataset")
reviews = pipeline.reviews[:-1000] # dont take validation set from training
reviews = [reviews[i] for i in inds] # get the subset that encodings correspond to

# Get HDBSCAN labels 
def get_labels(reduce_dim = 2): # set reduce_dim to -1 for no reduction
    print("Performing UMAP reduction...")
    tform = UMAP(
        n_neighbors = 30,
        min_dist = 0.0,
        n_components = reduce_dim,
        metric = 'cosine',
        random_state = 42,
        low_memory = False
    ).fit(R)
    
    z = tform.transform(R)

    print("Clustering...")
    clusterer = HDBSCAN(
        min_samples = 10,
        min_cluster_size = 10, # min number of samples per label we'll take
    ).fit(z)

    return clusterer.labels_, clusterer, tform

# Return list of indices of all elements that are not a given label
def cull(labels, to_cull):
    return np.where(labels != to_cull)[0].tolist()

# Print samples from every label 
def print_samples(labels, samples_per = 15):
    min_label = labels[labels != -1].min()
    max_label = labels[labels != -1].max()

    for l in range(min_label, max_label + 1):
        where_l = np.where(labels == l)[0].tolist()

        print("================")
        print("LABEL: {}".format(l))
        l_reviews = [reviews[i] for i in where_l]
        random.shuffle(l_reviews)

        n_samples =  min(len(where_l), samples_per)
        for rev in reviews[:n_samples]:
            print(rev)

# Slight modification to above script
# gives user samples for a single label and has them handwrite a caption
# do this to generate captions for each label
def make_captions(labels, samples_per = 15):
    min_label = labels[labels != -1].min()
    max_label = labels[labels != -1].max()

    f = open("metalabel_data/user_captions.txt", 'w')

    for l in range(min_label, max_label + 1):
        where_l = np.where(labels == l)[0].tolist()

        print("================")
        print("LABEL: {}".format(l))

        # Randomly sample some reviews that match this label
        l_reviews = [reviews[i] for i in where_l]
        random.shuffle(l_reviews)

        n_samples =  min(len(where_l), samples_per)
        for rev in l_reviews[:n_samples]:
            print(rev)

        caption = input()
        if caption == "-1": break
        f.write(caption + "\n")
    
    f.close()

# Returns metrics of how good clustering was
# [how many unclassified?, how many in modal label?]
def get_label_error(labels):
    noise_rate = (labels == -1).sum() / len(labels)

    labels = labels[cull(labels, -1)]

    modal_label = mode(labels).mode[0]
    modal_rate = (labels == modal_label).sum() / len(labels)

    return [noise_rate, modal_rate]

# get each label sorted by its frequency
# also returns list of respective proportions
def sorted_label_frequencies(labels):
    labels = labels[cull(labels, -1)]

    min_label = labels.min()
    max_label = labels.max()

    # proportion of a label
    def prop(label):
        return (labels == label).sum() / len(labels)

    # kvps of each label and its proportion
    kvps = [(l, prop(l)) for l in range(min_label, max_label + 1)]
    kvps.sort(key = lambda kvp: kvp[1])
    kvps.reverse() # in descending order

    return [kvp[0] for kvp in kvps], [kvp[1] for kvp in kvps]

if __name__ == "__main__":
    d = 40

    try:
        labels = np.load("metalabel_data/meta_labels.npy")
        clusterer = load("metalabel_data/clusterer.joblib")
        tform = load("metalabel_data/umap.joblib")
    except:
        labels, clusterer, tform = get_labels(d)
        np.save("metalabel_data/meta_labels.npy", labels)
        dump(clusterer, "metalabel_data/clusterer.joblib")
        dump(tform, "metalabel_data/umap.joblib")

    print(get_label_error(labels))
    labels_desc, label_props = sorted_label_frequencies(labels)
    print(labels_desc)
    print(label_props)

    make_captions(labels)


