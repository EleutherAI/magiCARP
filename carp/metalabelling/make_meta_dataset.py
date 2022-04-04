import os

# Check all prereq files are available
try:
    assert os.path.isfile("vis_checkpoints/umap.joblib")
    assert os.path.isfile("vis_checkpoints/hdbscan.joblib")
    assert os.path.isfile("vis_checkpoints/umap_review_embeddings.npy")
    assert os.path.isfile("vis_checkpoints/names_for_meta.txt")
except:
    print("One or more pre-req files were not found")
    print("Ensure you run carp.examples.vis.meta_prep")
    exit()

from joblib import dump, load
import torch
import numpy as np

from carp.examples.vis.vis_util import scatter_with_names
import matplotlib.pyplot as plt

SENTIMENT_FILTER = True # filter out stories with positive sentiment?

embs = np.load("vis_checkpoints/umap_review_embeddings.npy")
clusterer = load("vis_checkpoints/hdbscan.joblib")
tform = load("vis_checkpoints/umap.joblib")
names = []
with open("vis_checkpoints/names_for_meta.txt") as f:
    names = f.readlines()
names = names[:len(embs)]

print("Pre-req files found.")

labels = clusterer.labels_

# cull noise labels
def cull(labels, to_cull):
    return np.where(labels != to_cull)[0].tolist()
labels = labels[cull(labels, -1)]

c = labels / labels.max()

# remove blindly positive reviews
from carp.metalabelling.sentiment_filtering import filter_on_sentiment

if SENTIMENT_FILTER:
    try:
        print("Loaded sentiments from previous run")
        inds = np.load("vis_checkpoints/sent_filter_inds.npy")
    except:
        print("Calculating sentiments...")
        inds = filter_on_sentiment(names)
        np.save("vis_checkpoints/sent_filter_inds.npy", inds)

    embs = embs[inds]
    names = [names[ind] for ind in inds]
    labels = labels[inds]
    c = c[inds]
    sent_inds = inds

def plot_emb(embs, c = None, s = 3):
    if c is None:
        c = ["black"] * len(embs)
    plt.scatter(embs[:,0], embs[:,1], c = c, s = s)

# compute labelwise means
# first partition embedding space based on labels
emb_partition = []
for i in range(labels.max() + 1):
    emb_partition.append(embs[np.where(labels==i)])

means = np.stack([np.mean(emb, axis = 0) for emb in emb_partition])

# show means relative to rest of data
#plot_emb(embs, c = c)
#plot_emb(means, s = 50)
#plt.show()
#plt.close()

# computing distances from centroids (means) for some arbitrary points set (N x 2)
# notation: N: number of data points, C: number of cluster labels
def compute_dist_from_means(x):
    mu = torch.from_numpy(means) # -> C x 2
    diffs = mu[None,:] - x[:,None,:]
    # -> N x C x 2
    diffs = diffs ** 2
    diffs = diffs.sum(2) # -> N x C
    # -> diffs[i,j] = sqr of distance of i-th point from j-th mean
    diffs = torch.softmax(diffs, dim = 1)
    return diffs

# Now we need to sample randomly (~10k pairs) from the dataset. Let's re-use some old code from visualization
from carp.configs import CARPConfig
from carp.pytorch.model.architectures import *
from carp.pytorch.data import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading Dataset")
pipeline = BaseDataPipeline(path="carp/dataset")
passages = pipeline.passages
reviews = pipeline.reviews

sample_size = 10000
sample_inds = torch.randperm(len(passages))[:sample_size]

passages = [passages[i] for i in sample_inds]
reviews = [reviews[i] for i in sample_inds]

print("Loading Model")
config = CARPConfig.load_yaml("configs/carp_cloob.yml")
cloob_model = CARPCloob(config.model)
cloob_model.load("checkpoints/CLOOB_CP/")
cloob_model = cloob_model.to(device)

tokenize = cloob_model.passage_encoder.call_tokenizer

# Want to chunk reviews
CHUNK_SIZE = 1024
N_CTX = 512

reviews = [review[-N_CTX:] for review in reviews]
umap_encs = []

print("Encoding reviews...")
for i in range(0, len(reviews), CHUNK_SIZE):
    r_batch = reviews[i:i+CHUNK_SIZE]
    r_tok_out = tokenize(r_batch)

    r_input_ids = r_tok_out["input_ids"]
    r_masks = r_tok_out["attention_mask"]
    r_input = BatchElement(r_input_ids, r_masks)

    with torch.no_grad():
        rev_encs = cloob_model.encode_reviews(r_input)
    rev_encs = rev_encs.hidden.cpu().numpy()
    rev_encs = tform.transform(rev_encs)
    umap_encs.append(rev_encs)

umap_encs = np.concatenate(umap_encs)

print("Getting Distances To Means...")
logits = compute_dist_from_means(torch.from_numpy(umap_encs)).numpy()

import pandas as pd

df = pd.DataFrame()
df["passages"] = passages
for i in range(logits.shape[1]):
    s = str(i)
    df[s] = logits[:,i]

df.to_csv("passage_metalabel_dataset.csv")