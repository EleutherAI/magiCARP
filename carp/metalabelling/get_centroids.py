from joblib import dump, load
import torch 
import numpy as np

from carp.pytorch.data import *

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

# Load dataset
print("Load Dataset...")
pipeline = BaseDataPipeline(path="carp/dataset")
reviews = pipeline.reviews[:-1000] # dont take validation set from training
reviews = [reviews[i] for i in inds] # get the subset that encodings correspond to

# get indexes for negative and positive reviews
neg_inds = filter_on_sentiment(reviews, lambda x: x < 0)
pos_inds = np.setdiff1d(
    np.arange(len(reviews)), neg_inds, # all inds - neg inds
    assume_unique = True
)

# for all labels
# get mean of positive and negative reviews under that label
# -> ([n_labels, latent_dim], [n_labels, latent_dim]) -> [2, n_labels, latent_dim]
#       for positive and negative means respectively
def label_means(labels):
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

    for l in range(min_label, max_label + 1):
        pos_where = np.where(pos_labels == l)[0].tolist()
        neg_where = np.where(neg_labels == l)[0].tolist()

        pos_l_encs = pos_encs[pos_where]
        neg_l_encs = neg_encs[neg_where]

        means_pos[l - min_label] = pos_l_encs.mean(0)
        means_neg[l - min_label] = neg_l_encs.mean(0)
    
    return torch.stack([means_pos, means_neg])


if __name__ == "__main__":
    labels = clusterer.labels_
    means = label_means(labels)
    torch.save(means, "metalabel_data/centroids.pt")
