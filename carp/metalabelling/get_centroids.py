from joblib import dump, load
import torch 
import numpy as np

from carp.pytorch.data import *
from carp.metalabelling.utils import cull

# Check all prereq files are available
try:
    tform = load("metalabel_data/umap.joblib")
    clusterer = load("metalabel_data/clusterer.joblib")
    review_encs = torch.load("metalabel_data/review_encs.pt")
    inds = torch.load("metalabel_data/embedding_inds.pt")
except:
    print("One or more pre-req files were not found")
    print("Ensure you run carp.metalabelling.encode_all_reviews")
    print("And carp.metalabelling.explore_clusters")
    exit()

from carp.metalabelling.sentiment_filtering import filter_on_sentiment

# for all labels
# get mean of positive and negative reviews under that label
# -> ([n_labels, latent_dim], [n_labels, latent_dim]) -> [2, n_labels, latent_dim]
#       for positive and negative means respectively
def label_means(labels, pos_inds, neg_inds, review_encs, mean_strategy = "euclidean"):
    # firstly get labels of all positive and negative samples
    pos_labels = labels[pos_inds]
    neg_labels = labels[neg_inds]

    pos_encs = review_encs[pos_inds]
    neg_encs = review_encs[neg_inds]
    _, latent_dim = pos_encs.shape

    min_label = labels[labels != -1].min()
    max_label = labels[labels != -1].max()

    n_labels = max_label - min_label + 1
    means_pos = torch.zeros(n_labels, latent_dim)
    means_neg = torch.zeros(n_labels, latent_dim)
    means = torch.zeros(n_labels, latent_dim)

    if mean_strategy == "euclidean":
        mean_fn = lambda x: x.mean(0)
    # maybe add cos mean later

    for l in range(min_label, max_label + 1):
        pos_where = np.where(pos_labels == l)[0].tolist()
        neg_where = np.where(neg_labels == l)[0].tolist()

        pos_l_encs = pos_encs[pos_where]
        neg_l_encs = neg_encs[neg_where]
        overall_l_encs = torch.cat([pos_l_encs, neg_l_encs])

        means_pos[l - min_label] = mean_fn(pos_l_encs)
        means_neg[l - min_label] = mean_fn(neg_l_encs)
        means[l - min_label] = mean_fn(overall_l_encs)
    
    return torch.stack([means, means_pos, means_neg])


# If captions for clusters exist, we can incorporate them
# labels with caption "?" become -1
# multiple labels with same caption get collapsed to first label
def use_captions(labels):
    with open("metalabel_data/user_captions.txt") as file:
        lines = file.readlines()
    lines = [line[:-1] for line in lines] # remove the \n at the end
    kvps = [[i, lines[i]] for i in range(len(lines))]

    # first get rid of the labels that dont have captions (i.e. ?)
    inds = []
    for i in range(len(kvps)):
        if kvps[i][1] == "?":
            inds.append(i)

    labels[inds] = -1
    kvps = [[-1 if i in inds else i, kvps[i][1]] for i in range(len(kvps))]

    # now gather labels that have multiple captions
    ind_groups = []
    processed = []
    for i in range(len(kvps)):
        if kvps[0] == "-1": continue
        if i in processed: continue

        group = [i]

        # check all preceeding kvps for same caption
        for j in range(i + 1, len(kvps)):
            if kvps[j][1] == kvps[i][1]:
                group.append(j)
                processed.append(j)
        ind_groups.append(group)

    # ind groups is now list of groups of indices
    # a single group represents multiple labels with same caption
    
    # now for each group, make labels the same as the first in the group
    for group in ind_groups:
        to_set = kvps[group[0]][0] # the label we want to set everything else to
        for ind in group[1:]:
            labels[labels == ind] = to_set
            kvps[ind][0] = to_set

    # want to convert all these labels into [-1, 0, ... n_labels - 1]
    unique_vals = np.unique(labels)
    map_ = {unique_vals[i] : i - 1 for i in range(len(unique_vals))}
    kvps = list(set(map(tuple, kvps))) # remove duplicates
    kvps = [[map_[kvps[i][0]], kvps[i][1]] for i in range(len(kvps))] # change labels to new range
    kvps.sort(key = lambda x: x[0]) # sort by label

    captions = [kvp[1] for kvp in kvps] # recover captions for new label set

    labels = np.vectorize(map_.get)(labels) # convert labels to new range

    return labels, captions


if __name__ == "__main__":
    labels = clusterer.labels_
    labels, captions = use_captions(labels)
    # this should create a lot of new -1 labels, let's cull based on this
    print(captions[1:])

    cull_inds = cull(labels, -1)
    labels = labels[cull_inds]
    inds = inds[cull_inds]
    review_encs = review_encs[cull_inds]

    print("Load Dataset...")
    pipeline = BaseDataPipeline(path="carp/dataset")
    reviews = pipeline.reviews[:-1000] # dont take validation set from training
    reviews = [reviews[i] for i in inds] # get the subset that encodings correspond to

    neg_inds = filter_on_sentiment(reviews, lambda x: x < 0.5)
    print(neg_inds)
    np.save("metalabel_data/neg_inds.npy", neg_inds)

    pos_inds = np.setdiff1d(
        np.arange(len(review_encs)), neg_inds, # all inds - neg inds
        assume_unique = True
    )

    means = label_means(labels, pos_inds, neg_inds, review_encs)
    torch.save(means, "metalabel_data/centroids.pt")
