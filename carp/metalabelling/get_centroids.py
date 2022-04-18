from joblib import dump, load
import torch 
import numpy as np

from carp.pytorch.data import *
from carp.metalabelling.utils import cull, nsphere_centroid 

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
from carp.metalabelling.utils import use_captions
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
    if mean_strategy == "angular":
        mean_fn = nsphere_centroid

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

if __name__ == "__main__":
    labels = clusterer.labels_
    labels, captions = use_captions(labels)
    # this should create a lot of new -1 labels, let's cull based on this

    cull_inds = cull(labels, -1)
    labels = labels[cull_inds]
    inds = inds[cull_inds]
    review_encs = review_encs[cull_inds]

    print("Load Dataset...")
    pipeline = BaseDataPipeline(path="carp/dataset")
    reviews = pipeline.reviews[:-1000] # dont take validation set from training
    reviews = [reviews[i] for i in inds] # get the subset that encodings correspond to

    neg_inds = filter_on_sentiment(reviews, lambda x: x < 0.5)
    np.save("metalabel_data/neg_inds.npy", neg_inds)

    pos_inds = np.setdiff1d(
        np.arange(len(review_encs)), neg_inds, # all inds - neg inds
        assume_unique = True
    )

    means = label_means(labels, pos_inds, neg_inds, review_encs)
    torch.save(means, "metalabel_data/centroids.pt")
