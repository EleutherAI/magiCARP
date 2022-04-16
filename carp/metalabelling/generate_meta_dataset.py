import os
import torch
import torch.nn.functional as f
import numpy as np
import joblib
import hdbscan

from carp.metalabelling.encode_all_reviews import enc_reviews
from carp.pytorch.data import *

# check if centroids have been created
try:
    centroids = torch.load("metalabel_data/centroids.pt").half() # [3, k, d], overall, pos and neg centroids
except:
    print("One or more pre-req files were not found")
    print("Ensure you run carp.metalabelling.get_centroids")

# Load dataset
print("Load Dataset...")
pipeline = BaseDataPipeline(path="carp/dataset")
passages = pipeline.passages[:-1000] # dont take validation set from training

# We observed that previously, ~50% of reviews were unlabelled by HDBSCAN
# Assuming a random sample of (passage, review) pairs, and assuming reviews used
# for metalabel generation were randomly sampled, we can expect
# ~50% of (passage, review) pairs will have a pseudo-label
remove_unclassified = False

n_samples = 50000

# if we're gonna get 50%, take double to still get around above count
if remove_unclassified:
    n_samples *= 2

# try to load review encs and their indices
try:
    review_encs = torch.load("metalabel_data/meta_review_encs.pt")
    inds = torch.load("metalabel_data/meta_inds.pt")
    assert len(review_encs) == n_samples
except:
    # make the random state different from what the review embeddings from centroid step were generated with
    enc_reviews(n_samples, force_fresh = True, ind_path = "metalabel_data/meta_inds.pt", enc_path = "metalabel_data/meta_review_encs.pt", random_state = 1)
    review_encs = torch.load("metalabel_data/meta_review_encs.pt").half()
    inds = torch.load("metalabel_data/meta_inds.pt")

passages = [passages[ind] for ind in inds]

if remove_unclassified:
    try:
        clusterer = joblib.load("metalabel_data/clusterer.joblib")
    except:
        print("Trying to remove unclassified reviews except clusterer hasn't been run yet")
        print("Run carp.metalabelling.expore_cluster")

    # Return list of indices of all elements that are not a given label
    def cull(labels, to_cull):
        return np.where(labels != to_cull)[0].tolist()
    
    labels = hdbscan.approximate_predict(clusterer, review_encs)
    unclass_inds = cull(labels, -1) # remove noise points

    passages = [passages[i] for i in unclass_inds]
    review_encs = [review_encs[i] for i in unclass_inds]

    print("Removed {} unclassified reviews".format(len(unclass_inds)))

import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get classifier distribution from centroids
# gets cosine similarity for each sample with each centroid
def get_cos_sims(review_encs, centroids):
    # review_encs: [n, d]
    # centroids: [k, d]
    review_encs = review_encs.to(device) # cuda speeds up the matmult
    centroids = centroids.to(device)

    centroid_norms = torch.norm(centroids, dim = 1) # [k]
    review_norms = torch.norm(review_encs, dim = 1) # [n]

    norm_scale = centroid_norms[None,:] * review_norms[:,None] # [n, k]

    cos_sims = review_encs @ centroids.t() / norm_scale # [n, k]

    return cos_sims.cpu()

import pandas as pd

# split l into chunks of given size
def chunk(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

# to bypass memory errors, write each column separately
def make_csv_dataset(review_encs, centroids, mode = "pos"):
    df = pd.DataFrame()
    df["passages"] = passages

    if mode == "overall":
        means = centroids[0]
    elif mode == "pos":
        means = centroids[0]
    else:
        means = centroids[1]

    n_labels = len(means)

    chunk_size = 1024 * 8

    rev_chunks = chunk(review_encs, chunk_size)
    probs = torch.cat([get_cos_sims(rev, means) for rev in rev_chunks])
    #for i in range(probs.shape[1]):
    #    s = str(i)
    #    df[s] = probs[:,i]

    df.to_csv("metalabel_data/dataset.csv")
    torch.save(probs, "metalabel_data/dataset_centroid_dists_{}.pt".format(mode))

if __name__ == "__main__":
    make_csv_dataset(review_encs, centroids, mode = "overall")